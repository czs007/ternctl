"""Salvage — recover Milvus 2.6 WAL messages directly from Kafka, starting from
the SalvageCheckpoint captured at force-promote time, decoded to JSON Lines.

Why read Kafka directly instead of milvus's DumpMessages gRPC API? That API has
known issues in 2.6.x; reading Kafka is straightforward once the wire format is
decoded correctly (this module does). Output is one JSON object per line with
header + body decoded; downstream code consumes the JSONL and decides how to
replay / archive / dedupe.

Start offset is resolved in priority order:
  --from-offset  >  --from-checkpoint-file  >  live GetReplicateInfo

The checkpoint-file path is RECOMMENDED — it works after force-promote, when the
live GetReplicateInfo API is unavailable (milvus-io/milvus#50344). Capture the
file with `ternctl force-promote --salvage-source-cluster-id ...`.

Ordering: a single pchannel (one Kafka topic, partition 0) is strictly ordered
by offset. Across pchannels, sort by the `time_tick` field (globally monotonic).
Transactional groups (BeginTxn → CommitTxn/AbortTxn) must be replayed as a unit.
"""
import base64
import copy
import json
import os
import time
from collections import Counter

import grpc
from pymilvus.grpc_gen import milvus_pb2, milvus_pb2_grpc, msg_pb2, schema_pb2

from .config import load_config
from .output import header, kv, info, warn, _green, _red, _yel, _cyan, _bold, _dim


# Source: pkg/proto/messages.proto enum MessageType in milvus/2.6.
MSG_TYPES = {
    0: "Unknown", 1: "TimeTick", 2: "Insert", 3: "Delete", 4: "Flush",
    5: "CreateCollection", 6: "DropCollection", 7: "CreatePartition",
    8: "DropPartition", 9: "ManualFlush", 10: "CreateSegment", 11: "Import",
    12: "SchemaChange", 13: "AlterCollection", 14: "AlterLoadConfig",
    15: "DropLoadConfig", 16: "CreateDatabase", 17: "AlterDatabase",
    18: "DropDatabase", 19: "AlterAlias", 20: "DropAlias", 21: "RestoreRBAC",
    22: "AlterUser", 23: "DropUser", 24: "AlterRole", 25: "DropRole",
    26: "AlterUserRole", 27: "DropUserRole", 28: "AlterPrivilege",
    29: "DropPrivilege", 30: "AlterPrivilegeGroup", 31: "DropPrivilegeGroup",
    32: "AlterResourceGroup", 33: "DropResourceGroup", 34: "CreateIndex",
    35: "AlterIndex", 36: "DropIndex", 37: "FlushAll", 38: "TruncateCollection",
    700: "AlterWAL",
    800: "AlterReplicateConfig",   # fence message
    900: "BeginTxn", 901: "CommitTxn", 902: "AbortTxn",  # transaction markers
}


def _safe_proto_parse(proto_cls, raw):
    try:
        m = proto_cls()
        m.ParseFromString(raw)
        return m
    except Exception:
        return None


def _field_data_summary(field_data_list):
    """Summarize each field instead of enumerating per-row bytes."""
    out = []
    for fd in field_data_list:
        item = {"field_name": fd.field_name, "field_id": fd.field_id,
                "type": schema_pb2.DataType.Name(fd.type) if fd.type else "Unknown"}
        if fd.HasField("scalars"):
            sc = fd.scalars
            for k in ("long_data", "int_data", "string_data", "bool_data",
                      "float_data", "double_data", "json_data", "bytes_data",
                      "array_data"):
                if sc.HasField(k):
                    arr = getattr(getattr(sc, k), "data", None)
                    if arr is not None:
                        item["row_count"] = len(arr); break
        elif fd.HasField("vectors"):
            v = fd.vectors
            item["dim"] = v.dim
            for k in ("float_vector", "binary_vector", "float16_vector",
                      "bfloat16_vector", "sparse_float_vector"):
                if v.HasField(k):
                    item["vector_kind"] = k; break
        out.append(item)
    return out


def _decode_header_b64(raw):
    """The specialized header (Kafka header `_h`) is shipped as base64; decode it
    against milvus's internal messagespb proto on the receiving side (those types
    are not in pymilvus.grpc_gen)."""
    if not raw:
        return None
    return base64.b64encode(raw).decode()


def _decode_body(type_id, raw):
    """Decode body (Kafka Value bytes). Insert/Delete bodies are
    msg_pb2.InsertRequest / DeleteRequest. Other types fall through to raw."""
    if not raw:
        return None
    mapping = {2: msg_pb2.InsertRequest, 3: msg_pb2.DeleteRequest}
    cls = mapping.get(type_id)
    if cls is None:
        return {"raw_b64_len": len(raw)}
    parsed = _safe_proto_parse(cls, raw)
    if parsed is None:
        return {"decode_error": True, "raw_size": len(raw)}
    out = {}
    if type_id == 2:   # Insert
        out["collection_name"] = parsed.collection_name
        out["partition_name"]  = parsed.partition_name
        out["collectionID"]    = parsed.collectionID
        out["partitionID"]     = parsed.partitionID
        out["segmentID"]       = parsed.segmentID
        out["num_rows"]        = parsed.num_rows
        out["rowID_count"]     = len(parsed.rowIDs)
        out["timestamp_count"] = len(parsed.timestamps)
        out["fields_summary"]  = _field_data_summary(parsed.fields_data)
    elif type_id == 3:  # Delete
        out["collection_name"] = parsed.collection_name
        out["partition_name"]  = parsed.partition_name
        out["collectionID"]    = parsed.collectionID
        out["partitionID"]     = parsed.partitionID
        out["segmentID"]       = parsed.segment_id
        out["num_rows"]        = parsed.num_rows
        if parsed.HasField("primary_keys"):
            pk = parsed.primary_keys
            if pk.HasField("int_id"):
                out["pk_count"] = len(pk.int_id.data)
                out["pk_sample"] = list(pk.int_id.data[:5])
            elif pk.HasField("str_id"):
                out["pk_count"] = len(pk.str_id.data)
                out["pk_sample"] = list(pk.str_id.data[:5])
    return out


def get_salvage_checkpoint(new_primary_uri, source_cluster_id, target_pchannel,
                           token="root:Milvus"):
    """Call GetReplicateInfo on the new primary; return (kafka_offset, time_tick)
    or (None, None). NOTE: broken on an independent primary post-force-promote
    (milvus-io/milvus#50344) — prefer --from-checkpoint-file."""
    host_port = new_primary_uri.replace("http://", "").replace("https://", "").rstrip("/")
    channel = grpc.insecure_channel(host_port)
    stub = milvus_pb2_grpc.MilvusServiceStub(channel)
    md = [("authorization", base64.b64encode(token.encode()).decode())]
    try:
        resp = stub.GetReplicateInfo(
            milvus_pb2.GetReplicateInfoRequest(
                source_cluster_id=source_cluster_id,
                target_pchannel=target_pchannel),
            metadata=md, timeout=15)
    finally:
        channel.close()
    sc = resp.salvage_checkpoint
    if not sc or not sc.message_id:
        return None, None
    # MessageID is the kafka offset, base-36 encoded by milvus
    # (pkg/streaming/util/message/encoder.go, base = 36).
    raw = sc.message_id.id
    s = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
    try:
        offset = int(s, 36)
    except Exception as e:
        raise RuntimeError(f"can't decode salvage checkpoint message_id ({s!r}): {e}")
    return offset, sc.time_tick


def load_checkpoint_file(path, pchannel):
    """Resolve a start offset from a salvage checkpoint JSON produced by
    `ternctl force-promote --salvage-source-cluster-id`. Matches `pchannel`
    against each entry's source_pchannel_topic / target_pchannel.

    Returns (start_offset, time_tick, source_cluster_id, matched_entry).
    """
    with open(path) as fh:
        doc = json.load(fh)
    src_cluster = doc.get("source_cluster_id")
    entries = doc.get("pchannels", [])
    if not entries:
        raise RuntimeError(f"{path}: no 'pchannels' array — not a salvage checkpoint file")

    match = None
    for e in entries:
        if pchannel in (e.get("source_pchannel_topic"), e.get("target_pchannel")):
            match = e
            break
    if match is None:
        known = ", ".join(
            e.get("source_pchannel_topic") or e.get("target_pchannel") or "?"
            for e in entries[:6])
        raise RuntimeError(
            f"{path}: no entry matches --source-pchannel {pchannel!r}. "
            f"Known pchannels: {known}{'…' if len(entries) > 6 else ''}")

    if match.get("status") != "ok":
        raise RuntimeError(
            f"{path}: checkpoint for {pchannel!r} has status={match.get('status')!r}, "
            f"not 'ok'. No usable offset was captured — it may have had no CDC progress "
            f"at prefetch time, or the prefetch RPC failed. You can still dump from the "
            f"topic start with --from-offset 0 (and dedupe downstream).")
    off = match.get("kafka_offset")
    if off is None:
        raise RuntimeError(
            f"{path}: entry for {pchannel!r} has no kafka_offset (non-Kafka MQ?). "
            f"This salvage path only supports Kafka.")
    return off + 1, match.get("time_tick"), src_cluster, match


def do_salvage(args):
    """Entry point: single-pchannel dump, or a sweep over ALL pchannels of
    --source-cluster (one output file per pchannel)."""
    try:
        from kafka import KafkaConsumer, TopicPartition  # noqa: F401 — fail early
    except ImportError:
        raise RuntimeError(
            "salvage needs kafka-python — install it with: pip install 'ternctl[salvage]'")

    # --source-cluster pulls kafka_brokers (and the pchannel layout) from the
    # config file, so a sweep needs nothing but the checkpoint file.
    entry = {}
    if args.source_cluster:
        cfgmap = load_config(getattr(args, "config", None))
        if args.source_cluster not in cfgmap:
            known = ", ".join(sorted(cfgmap)) or "(empty)"
            raise RuntimeError(
                f"--source-cluster '{args.source_cluster}' is not in the config "
                f"file. known clusters: {known}")
        entry = cfgmap[args.source_cluster]
        if not args.kafka_brokers:
            args.kafka_brokers = entry.get("kafka_brokers")
        if not args.source_cluster_id:
            args.source_cluster_id = args.source_cluster
    if not args.kafka_brokers:
        raise RuntimeError(
            "--kafka-brokers is required (or set kafka_brokers on the "
            "--source-cluster's config entry: ternctl config add <name> "
            "--kafka-brokers host:9092)")

    if args.source_pchannel:
        _salvage_one(args)
        return

    # === Sweep mode: all pchannels of --source-cluster
    if not args.source_cluster:
        raise RuntimeError(
            "give --source-pchannel TOPIC for a single dump, or "
            "--source-cluster NAME to sweep all of its pchannels")
    n = int(entry.get("pchannel_num", 16))
    outdir = None
    if not args.summary_only:
        outdir = args.output_dir or f"salvage_{args.source_cluster}"
        os.makedirs(outdir, exist_ok=True)
    results = []
    for i in range(n):
        sub = copy.copy(args)
        sub.source_pchannel = f"{args.source_cluster}-rootcoord-dml_{i}"
        if outdir is not None:
            sub.output = os.path.join(outdir, f"salvage_dml_{i}.jsonl")
        try:
            written = _salvage_one(sub)
            results.append((sub.source_pchannel, written, None))
        except RuntimeError as e:
            warn(f"{sub.source_pchannel}: {e}")
            results.append((sub.source_pchannel, 0, str(e)))
    header("SWEEP SUMMARY", f"{n} pchannels of {args.source_cluster}")
    total = 0
    for pch, w, err in results:
        mark = (_red("✗ " + err[:70]) if err
                else (_green(f"{w} msgs") if w else _dim("0 msgs")))
        print(f"  {pch:46} {mark}")
        total += w
    print()
    kv("total recovered", str(total), _bold)
    if outdir is not None:
        kv("output dir", os.path.abspath(outdir), _green)
    info("replay MUST dedupe by primary key — the dump is a superset of the "
         "lost data by design (see HANDOFF §3.6)")


def _salvage_one(args):
    """Drain the salvage window of ONE pchannel from Kafka into JSON Lines.
    Returns the number of messages recovered."""
    from kafka import KafkaConsumer, TopicPartition

    if not args.summary_only and not args.output:
        raise RuntimeError("--output is required unless --summary-only is set")
    live_lookup = (args.from_checkpoint_file is None and args.from_offset is None)
    if live_lookup and not (args.new_primary_uri and args.source_cluster_id):
        raise RuntimeError(
            "--new-primary-uri and --source-cluster-id are required unless "
            "--from-checkpoint-file or --from-offset is given")

    src_label = args.source_cluster_id or "(from checkpoint file)"
    mode_str = "SUMMARY ONLY (no jsonl written)" if args.summary_only else f"output → {args.output}"
    header("SALVAGE: KAFKA WAL DUMP",
           f"source:      {_cyan(src_label)}\n"
           f"pchannel:    {_cyan(args.source_pchannel)}\n"
           f"mode:        {mode_str}")

    # === Step 1: resolve the start offset
    if args.from_offset is not None:
        start_offset = args.from_offset
        kv("--from-offset override", f"start at offset {start_offset} (no checkpoint lookup)", _yel)
    elif args.from_checkpoint_file is not None:
        print(f"  {_dim('1/3')} reading prefetched salvage checkpoint from file…")
        start_offset, cp_tt, file_src, entry = load_checkpoint_file(
            args.from_checkpoint_file, args.source_pchannel)
        kv("checkpoint file", args.from_checkpoint_file, _dim)
        kv("source cluster", file_src or "?", _green)
        kv("salvage_checkpoint", f"kafka offset={entry['kafka_offset']}, time_tick={cp_tt}", _green)
        kv("start consuming at", f"offset={start_offset} (= checkpoint + 1)", _bold)
    else:
        print(f"  {_dim('1/3')} fetching salvage checkpoint from new primary (live)…")
        try:
            cp_offset, cp_tt = get_salvage_checkpoint(args.new_primary_uri,
                                                      args.source_cluster_id,
                                                      args.source_pchannel,
                                                      args.token)
            if cp_offset is None:
                warn("no salvage checkpoint present on the new primary — fallback to topic earliest")
                start_offset = 0
            else:
                start_offset = cp_offset + 1
                kv("salvage_checkpoint", f"kafka offset={cp_offset}, time_tick={cp_tt}", _green)
                kv("start consuming at", f"offset={start_offset} (= checkpoint + 1)", _bold)
        except Exception as e:
            warn(f"GetReplicateInfo failed ({type(e).__name__}) — this is the known "
                 f"post-force-promote breakage (milvus-io/milvus#50344).")
            warn("Prefer the prefetch path: capture a checkpoint at force-promote time "
                 "with `ternctl force-promote --salvage-source-cluster-id`, then pass it "
                 "here via --from-checkpoint-file. Falling back to topic earliest.")
            start_offset = 0
    print()

    # === Step 2: connect to Kafka, assign, seek
    print(f"  {_dim('2/3')} connecting to Kafka and seeking to start offset…")
    kafka_kwargs = {
        "bootstrap_servers": args.kafka_brokers,
        "enable_auto_commit": False,
        "consumer_timeout_ms": args.timeout_seconds * 1000,
        "max_partition_fetch_bytes": 64 * 1024 * 1024,
    }
    if args.kafka_sasl_mechanism:
        kafka_kwargs["sasl_mechanism"] = args.kafka_sasl_mechanism
        kafka_kwargs["sasl_plain_username"] = args.kafka_sasl_user
        kafka_kwargs["sasl_plain_password"] = args.kafka_sasl_password
        kafka_kwargs["security_protocol"] = "SASL_SSL" if args.kafka_ssl else "SASL_PLAINTEXT"
    elif args.kafka_ssl:
        kafka_kwargs["security_protocol"] = "SSL"

    consumer = KafkaConsumer(**kafka_kwargs)
    tp = TopicPartition(args.source_pchannel, 0)
    consumer.assign([tp])
    consumer.seek(tp, start_offset)
    print()

    # === Step 3: drain into jsonl
    print(f"  {_dim('3/3')} draining (cap: {args.max_msgs}, idle stop after {args.timeout_seconds}s)…")
    types_count = Counter()
    earliest_tt = latest_tt = earliest_offset = latest_offset = None
    written = 0
    t_start = time.time()

    f = open(args.output, "w") if not args.summary_only else None
    try:
        for msg in consumer:
            headers = {k: v for (k, v) in (msg.headers or [])}

            def hget(key):
                v = headers.get(key)
                if v is None:
                    return None
                return v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else v

            def hgetb(key):
                v = headers.get(key)
                if v is None:
                    return None
                return bytes(v) if isinstance(v, (bytes, bytearray)) else v.encode()

            # _t (MessageType enum): base-10
            type_str = hget("_t") or "0"
            try:
                type_id = int(type_str)
            except ValueError:
                type_id = -1
            type_name = MSG_TYPES.get(type_id, f"Type_{type_id}")

            # _tt (TimeTick), _wt (WAL term): base-36 via milvus's EncodeInt64
            tt_str = hget("_tt") or "0"
            try:
                tt = int(tt_str, 36)
            except ValueError:
                tt = 0
            wt_str = hget("_wt") or "0"
            try:
                wt = int(wt_str, 36)
            except ValueError:
                wt = 0

            if f is not None:
                rec = {
                    "offset": msg.offset,
                    "partition": msg.partition,
                    "kafka_ts_ms": msg.timestamp,
                    "type": type_name,
                    "type_id": type_id,
                    "time_tick": tt,
                    "vchannel": hget("_vc"),
                    "wal_term": wt,
                    "last_confirmed_id": hget("_lc"),
                    "header_b64": _decode_header_b64(hgetb("_h")),
                    "body_summary": _decode_body(type_id, msg.value),
                    "body_size": len(msg.value) if msg.value else 0,
                    "body_b64": base64.b64encode(msg.value).decode() if msg.value else None,
                }
                f.write(json.dumps(rec) + "\n")
            written += 1
            types_count[type_name] += 1
            if earliest_tt is None or tt < earliest_tt: earliest_tt = tt
            if latest_tt is None or tt > latest_tt: latest_tt = tt
            if earliest_offset is None: earliest_offset = msg.offset
            latest_offset = msg.offset
            if written >= args.max_msgs:
                break
    finally:
        if f is not None:
            f.close()
    consumer.close()
    elapsed = time.time() - t_start

    # === Summary
    header("RECOVERED", f"{written} messages in {elapsed:.1f}s")
    if written == 0:
        warn("no messages after the salvage checkpoint — nothing to recover from this pchannel")
    else:
        kv("offset range", f"{earliest_offset} → {latest_offset}", _bold)
        kv("time_tick range", f"{earliest_tt} → {latest_tt}", _bold)
        print(f"  {_dim('type breakdown:')}")
        for t, n in types_count.most_common():
            print(f"    {n:>6}  {_cyan(t)}")
        print()
        if args.summary_only:
            info("summary-only mode — no jsonl written. Re-run without --summary-only "
                 "to dump full bodies for replay.")
        else:
            kv("output", args.output, _green)
            jq_cmd = "jq 'select(.type==\"Insert\")' " + args.output
            info("each line: type, time_tick, vchannel, header_b64 (raw), body_summary, body_b64 (raw).")
            info(f"filter by type:            {_bold(jq_cmd)}")
            info("sort cross-pchannel order: by time_tick field (globally monotonic)")
            info("decode header_b64 against milvus's internal messagespb proto on your side.")

    if written and "BeginTxn" in types_count:
        print()
        warn("BeginTxn markers present — replay must respect transaction boundaries "
             "(pair each BeginTxn with its CommitTxn / AbortTxn).")
    return written
