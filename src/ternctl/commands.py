"""Subcommand implementations: do_rebuild / switchover / force_promote / ..."""
import datetime
import json
import os
import sys
import time

import grpc
from pymilvus.grpc_gen import common_pb2, milvus_pb2

from .output import (header, step, done, info, warn, kv, _green, _red, _yel, _cyan, _dim, _bold)
from .cluster import pchannels_of, auth_metadata, grpc_addr
from .config import load_config, load_defaults, save_config, resolve_cluster, config_path
from .replication import (build_replicate_config, apply_replicate_config,
                          independent_replicate_config, get_replicate_checkpoints, fetch_cdc_latency,
                          prefetch_salvage_checkpoints, call_with_retry,
                          RPC_RETRIES, RPC_TIMEOUT)
from .backup import (backup_create, restore_secondary, restore_backup,
                     backup_list_names, backup_get_info, run_backup_capture,
                     backup_source_cluster)
from .verify import verify


# --------------------------------------------------------------------------- #
def merged_replicate_config(upstream, downstream):
    """The upstream's CURRENT topology plus the new upstream→downstream edge,
    as one declarative config. UpdateReplicateConfiguration REPLACES the whole
    replicate state, so adding a second downstream (a→c while a→b exists) must
    carry the existing edges and cluster defs — a bare [a,c]/[a→c] config
    would silently tear down a→b. Existing cluster defs other than the two
    endpoints are kept verbatim; the endpoints' defs are rebuilt fresh (their
    uri/token may have changed)."""
    config = load_config(None)
    keep_clusters, edges = [], []
    try:
        cur = get_replicate_view(upstream)
        for c in cur.clusters:
            if c.cluster_id in (upstream.cluster_id, downstream.cluster_id):
                continue
            # GetReplicateConfiguration REDACTS credentials, so sending a
            # carried def back verbatim trips server-side validation
            # ("connection_param.token cannot be changed"). Rebuild carried
            # defs from our own config file whenever we know the cluster.
            if c.cluster_id in config:
                keep_clusters.append(
                    resolve_cluster("peer", c.cluster_id, config).milvus_cluster())
            else:
                keep_clusters.append(c)
                warn(f"carrying cluster '{c.cluster_id}' verbatim from the live "
                     f"view — its token is redacted there, so the apply may be "
                     f"rejected. Add it to the config file first: "
                     f"ternctl config add {c.cluster_id} --uri ... [--token ...]")
        edges = [(t.source_cluster_id, t.target_cluster_id)
                 for t in cur.cross_cluster_topology]
    except RuntimeError:
        pass  # no readable view (fresh/independent upstream) → start clean
    new_edge = (upstream.cluster_id, downstream.cluster_id)
    if new_edge not in edges:
        edges.append(new_edge)
    if len(edges) > 1:
        info("carrying existing topology: "
             + ", ".join(f"{s}→{t}" for (s, t) in edges if (s, t) != new_edge))
    return common_pb2.ReplicateConfiguration(
        clusters=list(keep_clusters) + [upstream.milvus_cluster(),
                                        downstream.milvus_cluster()],
        cross_cluster_topology=[
            common_pb2.CrossClusterTopology(source_cluster_id=s,
                                            target_cluster_id=t)
            for (s, t) in edges],
    )


def _milvus_client(cluster):
    from pymilvus import MilvusClient
    return MilvusClient(uri=cluster.uri, token=cluster.token)


def _overlapping_nonempty(upstream, downstream):
    """[(name, target_rows)] for source collections that already exist
    NON-EMPTY on the target. restore APPENDS into existing collections, so
    rebuilding over these duplicates every row (observed: a dirty standby
    ended up with 3x the source's data)."""
    src_cols = set(_milvus_client(upstream).list_collections())
    dst = _milvus_client(downstream)
    out = []
    for name in dst.list_collections():
        if name in src_cols:
            rows = int(dst.get_collection_stats(name).get("row_count", 0))
            if rows > 0:
                out.append((name, rows))
    return out


def merged_replicate_config_minus(upstream, exclude_id):
    """The upstream's current topology with ONLY the upstream→exclude_id edge
    removed — other downstream edges survive. Cluster defs are kept only for
    clusters still participating in a remaining edge (an edge-less def counts
    as a primary and trips the 'primary count is not 1' validator); carried
    defs are rebuilt from the config file (the live view redacts tokens).
    Removing the LAST edge naturally degenerates to the independent config
    (clusters=[upstream], topology=[]) — the correct single-pair behavior."""
    config = load_config(None)
    cur = get_replicate_view(upstream)
    edges = [(t.source_cluster_id, t.target_cluster_id)
             for t in cur.cross_cluster_topology
             if not (t.source_cluster_id == upstream.cluster_id
                     and t.target_cluster_id == exclude_id)]
    participants = {upstream.cluster_id} | {c for e in edges for c in e}
    clusters = [upstream.milvus_cluster()]
    for c in cur.clusters:
        if c.cluster_id == upstream.cluster_id or c.cluster_id not in participants:
            continue
        if c.cluster_id in config:
            clusters.append(resolve_cluster("peer", c.cluster_id, config).milvus_cluster())
        else:
            clusters.append(c)
    return common_pb2.ReplicateConfiguration(
        clusters=clusters,
        cross_cluster_topology=[
            common_pb2.CrossClusterTopology(source_cluster_id=s, target_cluster_id=t)
            for (s, t) in edges],
    )


def do_rebuild(args, upstream, downstream):
    header("REBUILD",
           f"source: {_cyan(upstream.cluster_id)} ({upstream.uri})\n"
           f"target: {_cyan(downstream.cluster_id)} ({downstream.uri})")
    # Read-only guard, nothing has been mutated yet. ternctl NEVER touches the
    # target's data: emptying a dirty target is the operator's decision, made
    # outside this command. We only refuse to make it worse — restore APPENDS
    # into existing collections, so rebuilding over them duplicates every row.
    overlap = _overlapping_nonempty(upstream, downstream)
    if overlap:
        listed = ", ".join(f"{n} ({r} rows)" for n, r in overlap)
        raise RuntimeError(
            f"target {downstream.cluster_id} already holds data for source "
            f"collections: {listed}. restore APPENDS into existing collections "
            f"— rebuilding now would duplicate every row. rebuild requires an "
            f"EMPTY target; ternctl will not delete target data for you.")
    step("1/4", "snapshot primary"); backup_create(args); done()
    step("2/4", "register topology on target")
    # Merge, don't replace: keeps any existing edges (e.g. a→b) intact when
    # adding this one — see merged_replicate_config.
    up2down = merged_replicate_config(upstream, downstream)
    apply_replicate_config(downstream, up2down, _quiet=True); done()
    step("3/4", "restore snapshot to target"); restore_secondary(args, upstream, downstream); done()
    step("4/4", f"enable {upstream.cluster_id} → {downstream.cluster_id} replication")
    apply_replicate_config(upstream, up2down, _quiet=True); done()
    if args.verify:
        verify(args, upstream, downstream)
    else:
        header("DONE")
        info(f"writes flow {_cyan(upstream.cluster_id)} → {_cyan(downstream.cluster_id)} continuously via CDC")


def do_backup(args):
    """Snapshot a single cluster via milvus-backup — e.g. before reinstalling it.
    milvus-backup reads etcd + object storage directly, so this works even when
    the cluster's streaming layer is wedged (a stuck replication edge)."""
    config = load_config(getattr(args, "config", None))
    cluster = resolve_cluster("backup", args.cluster, config)
    header("BACKUP",
           f"cluster:  {_cyan(cluster.cluster_id)} ({cluster.uri})\n"
           f"name:     {args.backup_name}\n"
           "milvus-backup reads etcd + object storage — works even if the "
           "cluster's streaming layer is wedged")
    step("1/1", f"snapshot {cluster.cluster_id} via milvus-backup")
    backup_create(args)
    done()
    header("DONE")
    info(f"backup '{_bold(args.backup_name)}' created (workdir: {args.backup_workdir}).")
    info(f"safe to reinstall {_cyan(cluster.cluster_id)} now — restore later with "
         f"{_bold('ternctl backup restore')}, or rebuild a fresh standby from the current primary.")


def do_restore(args):
    """Restore a milvus-backup snapshot into a cluster (rollback / clone).
    With --restore-suffix, originals are untouched (restored into new collections)."""
    config = load_config(getattr(args, "config", None))
    cluster = resolve_cluster("cluster", args.cluster, config)
    suffix = getattr(args, "restore_suffix", None)
    header("RESTORE",
           f"cluster:  {_cyan(cluster.cluster_id)} ({cluster.uri})\n"
           f"backup:   {args.backup_name}"
           + (f"\nsuffix:   {suffix} (restores into NEW collections; originals untouched)"
              if suffix else "\ninto the original collections")
           + "\ntarget must be an INDEPENDENT primary — restore creates collections "
             "(needs primary) and bulk-imports (blocked on a replicating cluster)")
    step("1/1", f"restore '{args.backup_name}' into {cluster.cluster_id}")
    restore_backup(args)
    done()
    header("DONE")
    info(f"backup '{_bold(args.backup_name)}' restored into {_cyan(cluster.cluster_id)}"
         + (f" as <name>{suffix}." if suffix else "."))


def do_switchover(args, config):
    """Graceful role flip: make --target the primary (RPO = 0). The target's
    current primary is auto-discovered from the target's own replicate config.
    SIBLING standbys of the old primary are CARRIED to the new one — the
    config sent is the old primary's full topology re-rooted at the target
    (UpdateReplicateConfiguration is full-state replacement: a bare two-
    cluster config would tear every sibling edge down; observed live)."""
    target = resolve_cluster("target", args.target, config,
                             token=getattr(args, "token", None), pchannel_num=getattr(args, "pchannel_num", None))
    tview = get_replicate_view(target,
                               getattr(args, "rpc_retries", None),
                               getattr(args, "rpc_timeout", None))
    incoming = sorted({t.source_cluster_id for t in tview.cross_cluster_topology
                       if t.target_cluster_id == target.cluster_id})
    outgoing = sorted({t.target_cluster_id for t in tview.cross_cluster_topology
                       if t.source_cluster_id == target.cluster_id})
    if not incoming:
        if outgoing:
            raise RuntimeError(
                f"{target.cluster_id} is already the primary of its edge "
                f"({target.cluster_id} → {', '.join(outgoing)}) — nothing to do")
        raise RuntimeError(
            f"{target.cluster_id} has no incoming replication edge — nothing "
            f"to switch over (see `ternctl topology`; create an edge with "
            f"`ternctl rebuild`)")
    if len(incoming) > 1:
        raise RuntimeError(
            f"{target.cluster_id} reports multiple upstreams "
            f"({', '.join(incoming)}) — a pathological residual state; repair "
            f"it first (ternctl detach --downstream {target.cluster_id} "
            f"--upstream <extra-edge-source>), then switch over")
    up_cid = incoming[0]
    if up_cid not in config:
        raise RuntimeError(
            f"discovered current primary '{up_cid}' is not in the config file — "
            f"add it first: ternctl config add {up_cid} --uri ...")
    upstream = resolve_cluster("upstream", up_cid, config,
                               token=getattr(args, "token", None), pchannel_num=getattr(args, "pchannel_num", None))

    # Old primary's FULL topology, re-rooted at the target: every sibling
    # standby keeps replicating, now from the new primary (the CDC fence
    # rewires their source automatically when this config reaches them).
    uview = get_replicate_view(upstream,
                               getattr(args, "rpc_retries", None),
                               getattr(args, "rpc_timeout", None))
    siblings = sorted({t.target_cluster_id for t in uview.cross_cluster_topology
                       if t.source_cluster_id == upstream.cluster_id
                       and t.target_cluster_id != target.cluster_id})
    missing = [c for c in siblings if c not in config]
    if missing:
        raise RuntimeError(
            f"sibling standby(s) {', '.join(missing)} not in the config file — "
            f"their cluster defs must be rebuilt locally (the live view redacts "
            f"tokens). Add them first: ternctl config add <name> --uri ...")
    cluster_defs = [target.milvus_cluster(), upstream.milvus_cluster()] + [
        resolve_cluster("peer", c, config, token=getattr(args, "token", None),
                        pchannel_num=getattr(args, "pchannel_num", None)).milvus_cluster()
        for c in siblings]
    edges = [(target.cluster_id, upstream.cluster_id)] + [
        (target.cluster_id, c) for c in siblings]
    new_config = common_pb2.ReplicateConfiguration(
        clusters=cluster_defs,
        cross_cluster_topology=[
            common_pb2.CrossClusterTopology(source_cluster_id=s_,
                                            target_cluster_id=t_)
            for (s_, t_) in edges],
    )

    header("SWITCHOVER",
           f"target (new primary): {_cyan(target.cluster_id)} ({target.uri})\n"
           f"current primary:      {_cyan(upstream.cluster_id)} ({upstream.uri})")
    info(f"current primary auto-discovered from {target.cluster_id}'s "
         f"replicate config: {up_cid}")
    if siblings:
        info("carrying sibling standby(s) to the new primary: "
             + ", ".join(siblings))
    step("1/2", f"apply re-rooted topology to {upstream.cluster_id}")
    apply_replicate_config(upstream, new_config, _quiet=True); done()
    step("2/2", f"apply re-rooted topology to {target.cluster_id}")
    apply_replicate_config(target, new_config, _quiet=True); done()
    header("DONE")
    kv("new primary", f"{target.cluster_id} ({target.uri})", _green)
    kv("now standby", f"{upstream.cluster_id} ({upstream.uri})", _dim)
    if siblings:
        kv("still standby", ", ".join(f"{c} (now ← {target.cluster_id})"
                                      for c in siblings), _dim)
    info("point application writes at the new primary; old primary now receives via CDC")

def do_force_promote(args, target):
    # Pre-flight BEFORE the confirmation prompt: force-promote only applies to
    # a configured SECONDARY. Asking a human to solemnly confirm an operation
    # that is going to be refused anyway is ceremony in the wrong order.
    sources, view_ok = [], True
    try:
        cfg = get_replicate_view(target,
                                 getattr(args, "rpc_retries", None),
                                 getattr(args, "rpc_timeout", None))
        sources = sorted({t.source_cluster_id for t in cfg.cross_cluster_topology
                          if t.target_cluster_id == target.cluster_id})
        outgoing = sorted({t.target_cluster_id for t in cfg.cross_cluster_topology
                           if t.source_cluster_id == target.cluster_id})
        if not sources:
            if outgoing:
                raise RuntimeError(
                    f"{target.cluster_id} is a PRIMARY "
                    f"({target.cluster_id} → {', '.join(outgoing)}) — force-promote "
                    f"promotes a SECONDARY; nothing to do here")
            raise RuntimeError(
                f"{target.cluster_id} is INDEPENDENT (no replication edges) — "
                f"force-promote requires a configured secondary; nothing to promote")
    except RuntimeError as e:
        if "INDEPENDENT" in str(e) or "PRIMARY" in str(e):
            raise
        view_ok = False
        warn(f"could not read {target.cluster_id}'s replicate view ({e}) — "
             f"cannot verify it is a secondary; proceeding on your confirmation")

    # Salvage-source auto-discovery: the standby's own replicate config records
    # its incoming edge — its source IS the (dead) primary whose checkpoint we
    # must snapshot before the role flips. Opting OUT (--no-salvage) is the
    # explicit action, because skipping the prefetch makes the old primary's
    # in-flight data unrecoverable (see #50344 / HANDOFF §3.2).
    if not args.salvage_from and not getattr(args, "no_salvage", False):
        if not view_ok:
            warn("salvage-source auto-discovery unavailable (view unreadable) — "
                 "proceeding WITHOUT prefetch; pass --salvage-from "
                 "explicitly to capture one")
        if len(sources) == 1:
            args.salvage_from = sources[0]
            info(f"salvage source auto-discovered from {target.cluster_id}'s "
                 f"replicate config: {_cyan(sources[0])} "
                 f"(pass --no-salvage to skip the prefetch)")
        elif len(sources) > 1:
            warn(f"{target.cluster_id} has {len(sources)} incoming edges "
                 f"({', '.join(sources)}) — cannot pick a salvage source "
                 f"automatically, pass --salvage-from")
    if not getattr(args, "yes", False):
        ans = input(
            f"\nFORCE-PROMOTE will make '{target.cluster_id}' an independent primary.\n"
            f"Data written to the OLD primary after the CDC lag horizon will be LOST.\n"
            f"Type 'force-promote' to confirm: "
        ).strip()
        if ans != "force-promote":
            info("aborted")
            sys.exit(1)
    do_prefetch = bool(args.salvage_from)
    subtitle = (
        f"target: {_cyan(target.cluster_id)} ({target.uri})\n"
        f"{_yel('⚠')} RPO bounded by CDC lag at failure time — NOT zero"
    )
    if do_prefetch:
        subtitle += (
            f"\nprefetch salvage checkpoint from source "
            f"{_cyan(args.salvage_from)} before flipping role"
        )
    header("FORCE-PROMOTE", subtitle)

    # === Step 0 (optional): snapshot the salvage checkpoint while target is
    # still a secondary. GetReplicateInfo works in this state but breaks once
    # we force_promote — see milvus-io/milvus#50344.
    salvage_out_path = None
    if do_prefetch:
        n_step = "0/1"
        step(n_step, f"snapshot ReplicateCheckpoint for {target.pchannel_num} pchannels")
        pchannels = pchannels_of(target.cluster_id, target.pchannel_num)
        entries = prefetch_salvage_checkpoints(target, args.salvage_from, pchannels)
        ok = sum(1 for e in entries if e.get("status") == "ok")
        salvage_out_path = args.checkpoint_file or os.path.abspath(
            f"salvage_checkpoint_{target.cluster_id}_{int(time.time())}.json")
        try:
            ts_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        except AttributeError:
            ts_iso = datetime.datetime.utcnow().isoformat() + "Z"
        with open(salvage_out_path, "w") as f:
            json.dump({
                "version": 1,
                "prefetched_at_iso": ts_iso,
                "prefetched_at_unix": int(time.time()),
                "target_cluster_id": target.cluster_id,
                "target_uri": target.uri,
                "source_cluster_id": args.salvage_from,
                "pchannel_num": target.pchannel_num,
                "pchannels": entries,
                "_note": (
                    "ReplicateCheckpoint snapshot taken while target was still in "
                    "SECONDARY state, just before the force_promote RPC. Feed this "
                    "file into ternctl salvage --checkpoint-file to "
                    "recover messages the old primary's WAL retained but CDC didn't "
                    "forward in time. See milvus-io/milvus#50344 for context."
                ),
            }, f, indent=2)
        done(extra=f"{ok}/{len(pchannels)} ok → {os.path.basename(salvage_out_path)}")
        if ok < len(pchannels):
            rpc_err = sum(1 for e in entries if e.get("status", "").startswith("rpc_error"))
            empty   = sum(1 for e in entries if e.get("status") == "empty")
            if rpc_err == len(pchannels):
                warn(f"ALL {rpc_err} pchannels returned rpc_error — GetReplicateInfo is "
                     f"unreachable on this cluster. Likely cause: target is already a "
                     f"independent primary, see milvus-io/milvus#50344. Prefetch is only "
                     f"effective BEFORE force_promote takes effect.")
            elif rpc_err:
                warn(f"{rpc_err} pchannels rpc_error, {empty} empty — only "
                     f"{ok}/{len(pchannels)} have usable checkpoints.")
            else:
                warn(f"{empty} pchannel(s) returned empty — those have no CDC progress yet "
                     f"(brand-new pchannels, or replication just started).")

    # === Step 1: the actual force_promote
    step("1/1", f"promote {target.cluster_id} to independent primary")
    config = independent_replicate_config(target)
    apply_replicate_config(target, config, force_promote=True, _quiet=True); done()

    header("DONE")
    kv("new primary", f"{target.cluster_id} ({target.uri})  [INDEPENDENT — no standby]", _green)
    if salvage_out_path:
        kv("salvage snapshot", salvage_out_path, _bold)
        info(f"to recover from {args.salvage_from}'s WAL, feed it into:")
        info(f"  {_bold('ternctl salvage --checkpoint-file ' + salvage_out_path + ' --source-pchannel <topic> --kafka-brokers <hosts> --output salvage.jsonl')}")
    info(f"next: when the old primary recovers, run "
         f"{_bold('ternctl rebuild --upstream <new> --downstream <old>')} to re-establish a standby")


def do_status(args, upstream, downstream):
    header("STATUS",
           f"per-pchannel replication progress, reported by {_cyan(downstream.cluster_id)}\n"
           f"(source: {_cyan(upstream.cluster_id)})")
    # NOTE: target_pchannel param of GetReplicateInfo must be the TARGET's own
    # pchannel name (downstream.cluster_id prefix), NOT the source's. The
    # upstream version of milvus_dr.py uses upstream.cluster_id here and that
    # silently fails — RPCs hit milvus, hit the proxy handler, but the handler
    # cannot find a matching pchannel on the target and the request retries
    # forever until context-cancel (looks like a hang from the caller).
    pchannels = pchannels_of(downstream.cluster_id, downstream.pchannel_num)
    ticks = get_replicate_checkpoints(
        downstream, upstream.cluster_id, pchannels,
        rpc_tries=getattr(args, "rpc_retries", None) or RPC_RETRIES,
        rpc_timeout=getattr(args, "rpc_timeout", None) or RPC_TIMEOUT)
    if not ticks:
        warn("no replication info available (not started, or API unsupported)")
        return
    # Optional: real e2e replication lag, read straight from the source CDC
    # pod's /metrics (no Prometheus needed). {target_pchannel: avg_ms}.
    # --up-cdc overrides; otherwise use the upstream's cdc_metrics from config.
    cdc_url = getattr(args, "upstream_cdc_metrics", None) or upstream.cdc_metrics
    cdc_lat = fetch_cdc_latency(cdc_url) if cdc_url else None
    if cdc_url and cdc_lat is None:
        warn(f"could not read CDC metrics at {cdc_url} — showing progress only")

    active = idle = unreach = 0
    lat_vals = []
    for pch in pchannels:
        v = ticks.get(pch)
        if v is None:
            print(f"  {_red('○')} {pch:46} {_red('unreachable')} {_dim('(RPC failed)')}")
            unreach += 1
        elif v == 0:
            print(f"  {_yel('◌')} {pch:46} {_yel('idle')} {_dim('(configured, no traffic yet)')}")
            idle += 1
        else:
            tail = _dim("(data has flowed)")
            if cdc_lat is not None and pch in cdc_lat:
                ms = cdc_lat[pch]; lat_vals.append(ms)
                tail = (_green if ms < 1000 else _yel)(f"lag~{ms:.0f}ms") + _dim(" avg")
            print(f"  {_green('●')} {pch:46} {_green('active')} {tail}")
            active += 1
    print()
    parts = []
    if active:   parts.append(f"{active} active")
    if idle:     parts.append(f"{idle} idle")
    if unreach:  parts.append(f"{unreach} unreachable")
    summary = " / ".join(parts) + f"  (of {len(pchannels)})"
    if lat_vals:
        summary += f", avg lag {sum(lat_vals)/len(lat_vals):.0f}ms"
    if unreach:
        kv("summary", summary + " — unreachable channels need attention", _red)
    elif active and not idle:
        kv("summary", summary + " — all replicating ✓", _green)
    elif idle and not active:
        kv("summary", summary + " — edge configured but idle; write to the "
           "source to see checkpoints advance", _yel)
    else:
        kv("summary", summary, _yel)
    if active and cdc_lat is not None:
        info("lag = real CDC end-to-end latency (source→target per message, "
             "from the CDC pod's /metrics). It's a cumulative average since the "
             "CDC pod started, not a live window.")
    elif active:
        info("'active' = data has flowed; 'idle' = configured but none yet. "
             "Pass --upstream-cdc-metrics http://<source-cdc-pod>:9091 to also "
             "show real replication lag (e2e latency), or see Grafana's panel.")


def _cluster_line(name, e):
    """One formatted listing line for a config entry (shared by clusters / config list)."""
    extras = []
    if e.get("inter_uri"):     extras.append("inter=" + e["inter_uri"])
    if e.get("cdc_metrics"):   extras.append("cdc=" + e["cdc_metrics"])
    if e.get("backup_config"): extras.append("backup=" + os.path.basename(e["backup_config"]))
    if e.get("kafka_brokers"): extras.append("kafka=" + e["kafka_brokers"])
    if e.get("token"):         extras.append("token=set")
    return (f"  {_cyan(name):18} {_dim('uri=')}{e.get('uri','?')}"
            + ("  " + _dim(" ".join(extras)) if extras else ""))


def _grpc_ready(addr, timeout=2.0):
    """True if a plaintext gRPC channel to addr becomes READY within timeout.
    Transport-level only (TCP + HTTP/2), so it is NOT affected by the
    GetReplicateConfiguration hang on INDEPENDENT clusters — and fast."""
    ch = grpc.insecure_channel(addr)
    try:
        grpc.channel_ready_future(ch).result(timeout=timeout)
        return True
    except Exception:
        return False
    finally:
        ch.close()


def do_clusters(args):
    """List the clusters defined in the config file; --probe checks reachability."""
    cfg = load_config(getattr(args, "config", None))
    path = config_path(getattr(args, "config", None))
    header("CLUSTERS", f"{len(cfg)} configured in {path}")
    if not cfg:
        info("no clusters yet. Add one: "
             + _bold("ternctl config add <name> --uri http://...:19530"))
        return
    probe = getattr(args, "probe", False)
    for name in sorted(cfg):
        e = cfg[name]
        line = _cluster_line(name, e)
        if probe:
            t0 = time.time()
            ok = _grpc_ready(grpc_addr(e.get("uri", "")))
            ms = (time.time() - t0) * 1000
            line += "  " + (_green(f"✓ up {ms:.0f}ms") if ok else _red("✗ unreachable"))
        print(line)
    if not probe:
        info("add " + _bold("--probe") + " to check gRPC reachability of each uri")


def do_backups(args):
    """List the archive. With --cluster (and not --all), filter by each
    backup's TRUE source cluster, read from its raw meta in the archive —
    channel names embed the cluster id. Undeterminable backups (ghost meta,
    or empty-cluster snapshots with no channels) are listed, never hidden."""
    names = backup_list_names(args)
    cl = (getattr(args, "cluster", None) or "").split("=", 1)[0].strip() or None
    show_all = getattr(args, "all", False)
    cfg_name = os.path.basename(args.backup_config)

    if cl and not show_all:
        header("BACKUPS", f"source = {_cyan(cl)} — by recorded META (channel "
                          f"names), not by backup name. --all lists the whole "
                          f"shared archive (via {cfg_name})")
        matched, others, unattributed = [], [], []
        for n in names:
            src = backup_source_cluster(args, n)
            if src == cl:
                matched.append(n)
            elif src is None:
                unattributed.append(n)
            else:
                others.append((n, src))
        for n in matched:
            print(f"  {_cyan(n)}")
        if not matched:
            info(f"no backups recorded as taken from {cl}")
        if others:
            info(f"{len(others)} from other clusters (--all to list): "
                 + ", ".join(f"{n} ({s_})" for n, s_ in others[:3])
                 + (" …" if len(others) > 3 else ""))
        if unattributed:
            warn("source undeterminable (ghost meta, or snapshot of an empty "
                 "cluster): " + ", ".join(unattributed))
        return

    header("BACKUPS", f"whole archive via {_cyan(cfg_name)}"
                      + (" (--all)" if cl and show_all else ""))
    if not names:
        info("no backups found in this archive")
        return
    annotate = bool(cl and show_all)   # origins cost one meta read per backup
    if not getattr(args, "detail", False):
        for n in names:
            origin = ""
            if annotate:
                src = backup_source_cluster(args, n)
                origin = "  " + _dim(f"({src or 'source unknown'})")
            print(f"  {_cyan(n)}{origin}")
        info("add " + _bold("--detail") + " for size / milvus version / "
             "collections (one extra read per backup)")
        return
    for n in names:
        d = backup_get_info(args, n)
        if not d:
            print(f"  {_cyan(n):24} {_red('? could not read backup meta')}")
            continue
        cols = [c.get("collection_name", "?")
                for c in d.get("collection_backups") or []]
        print(f"  {_cyan(n):24} {_dim('size=')}{d.get('size', '?')}B"
              f"  {_dim('milvus=')}{d.get('milvus_version', '?')}"
              f"  {_dim('state=')}{d.get('state_code', '?')}"
              f"  {_dim('collections=')}{','.join(cols) or '-'}")

def get_replicate_view(cluster, rpc_tries=None, rpc_timeout=None):
    """One cluster's own replicate configuration (with the usual short-timeout
    retry against the INDEPENDENT-state hang). Raises RuntimeError if it stays
    unreachable after the retry budget."""
    def _call(timeout):
        stub, ch = cluster.stub()
        try:
            return stub.GetReplicateConfiguration(
                milvus_pb2.GetReplicateConfigurationRequest(),
                metadata=auth_metadata(cluster.token), timeout=timeout)
        finally:
            ch.close()
    try:
        return call_with_retry(_call, tries=rpc_tries or RPC_RETRIES,
                               timeout=rpc_timeout or RPC_TIMEOUT).configuration
    except grpc.RpcError as e:
        raise RuntimeError(
            f"GetReplicateConfiguration on {cluster.cluster_id} failed "
            f"({e.code().name if hasattr(e, 'code') else e})")


def discover_downstreams(args, upstream, config):
    """Resolved Cluster objects for every downstream in the upstream's own
    replicate config (unregistered ids are warned about and skipped)."""
    cfg = get_replicate_view(upstream,
                             getattr(args, "rpc_retries", None),
                             getattr(args, "rpc_timeout", None))
    targets = [t.target_cluster_id for t in cfg.cross_cluster_topology
               if t.source_cluster_id == upstream.cluster_id]
    out = []
    for tcid in targets:
        if tcid not in config:
            warn(f"downstream '{tcid}' is not in the config file — skipped. "
                 f"Add it: ternctl config add {tcid} --uri http://...:19530")
            continue
        out.append(resolve_cluster("downstream", tcid, config,
                                   token=getattr(args, "token", None),
                                   pchannel_num=getattr(args, "pchannel_num", None)))
    return targets, out


def for_each_downstream(args, upstream, config, fn):
    """--downstream omitted: discover the upstream's downstreams from its own
    replicate configuration and run fn(args, upstream, downstream) for each.
    Downstream URIs come from the config file — a discovered cluster_id that
    isn't configured there is reported and skipped. Returns the AND of fn's
    truthiness (None counts as ok) for verify-style exit codes."""
    cfg = get_replicate_view(upstream,
                             getattr(args, "rpc_retries", None),
                             getattr(args, "rpc_timeout", None))
    targets = [t.target_cluster_id for t in cfg.cross_cluster_topology
               if t.source_cluster_id == upstream.cluster_id]
    if not targets:
        warn(f"{upstream.cluster_id} has no outgoing replication edges — "
             f"nothing to do. (It is INDEPENDENT or itself a standby; "
             f"run `ternctl topology` to see the full picture.)")
        return True
    info(f"downstreams of {upstream.cluster_id} (from its replicate config): "
         + ", ".join(targets))
    all_ok = True
    for tcid in targets:
        if tcid not in config:
            warn(f"downstream '{tcid}' is not in the config file — skipped. "
                 f"Add it: ternctl config add {tcid} --uri http://...:19530")
            all_ok = False
            continue
        downstream = resolve_cluster("downstream", tcid, config,
                                     token=getattr(args, "token", None),
                                     pchannel_num=getattr(args, "pchannel_num", None))
        result = fn(args, upstream, downstream)
        all_ok = all_ok and (result is None or bool(result))
    return all_ok


def do_backup_get(args):
    """Show one backup's meta (backup get -n NAME)."""
    header("BACKUP", f"meta of {_cyan(args.backup_name)}")
    d = backup_get_info(args, args.backup_name)
    if not d:
        raise RuntimeError(
            f"could not read meta for '{args.backup_name}'. If `backup list` "
            f"warns 'can not read backup info' for it, this is a GHOST from an "
            f"interrupted create (binlogs written, meta never was) — "
            f"milvus-backup cannot delete those either; remove the S3 prefix "
            f"<backupRootPath>/{args.backup_name}/ yourself (see HANDOFF §2.2).")
    kv("name", d.get("name", "?"), _green)
    kv("source cluster", backup_source_cluster(args, args.backup_name) or "unknown")
    kv("state_code", str(d.get("state_code", "?")))
    kv("size", f"{d.get('size', '?')}B")
    kv("milvus_version", d.get("milvus_version", "?"))
    cols = [c.get("collection_name", "?") for c in d.get("collection_backups") or []]
    kv("collections", ", ".join(cols) or "-")


def do_backup_delete(args):
    """Delete one backup from the archive (backup delete -n NAME)."""
    header("BACKUP DELETE", _cyan(args.backup_name))
    if not getattr(args, "yes", False):
        ans = input(f"Delete '{args.backup_name}' from the archive? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            info("aborted")
            sys.exit(1)
    try:
        run_backup_capture(args, ["--config", args.backup_config,
                                  "delete", "-n", args.backup_name])
    except RuntimeError as e:
        msg = str(e)
        if "read backup meta" in msg or "does not exist" in msg:
            raise RuntimeError(
                f"milvus-backup cannot delete '{args.backup_name}' — its meta is "
                f"missing (ghost from an interrupted create). ternctl will not "
                f"touch S3 for you: remove the prefix "
                f"<backupRootPath>/{args.backup_name}/ in the backup bucket "
                f"yourself (see HANDOFF §2.2).") from e
        raise
    info(f"'{args.backup_name}' deleted from the archive")


def do_config(args):
    """Manage the cluster config file (~/.ternctl.yaml) — kubectl-config style."""
    cmd = args.config_command
    cfg = load_config(getattr(args, "config", None))
    path = config_path(getattr(args, "config", None))

    if cmd == "list":
        header("CONFIG", f"clusters in {path}")
        if not cfg:
            info("no clusters yet. Add one: "
                 + _bold("ternctl config add <name> --uri http://...:19530"))
            return
        for name in sorted(cfg):
            print(_cluster_line(name, cfg[name]))
        return

    if cmd == "show":
        import yaml
        print(yaml.safe_dump({"clusters": cfg}, sort_keys=True, default_flow_style=False)
              if cfg else "# (empty)")
        return

    if cmd == "add":
        entry = dict(cfg.get(args.name, {}))
        entry["uri"] = args.uri
        if args.inter is not None:        entry["inter_uri"] = args.inter
        if args.token is not None:        entry["token"] = args.token
        if args.pchannel_num is not None: entry["pchannel_num"] = args.pchannel_num
        if args.cdc_metrics is not None:  entry["cdc_metrics"] = args.cdc_metrics
        if getattr(args, "backup_config", None) is not None:
            entry["backup_config"] = os.path.abspath(args.backup_config)
        if getattr(args, "kafka_brokers", None) is not None:
            entry["kafka_brokers"] = args.kafka_brokers
        cfg[args.name] = entry
        saved = save_config(cfg, getattr(args, "config", None))
        header("CONFIG", f"saved '{_cyan(args.name)}' to {saved}")
        kv("uri", entry["uri"], _green)
        if entry.get("inter_uri"):     kv("inter_uri", entry["inter_uri"])
        if entry.get("cdc_metrics"):   kv("cdc_metrics", entry["cdc_metrics"])
        if entry.get("backup_config"): kv("backup_config", entry["backup_config"])
        if entry.get("kafka_brokers"): kv("kafka_brokers", entry["kafka_brokers"])
        return

    if cmd == "set-defaults":
        defaults = load_defaults(getattr(args, "config", None))
        if args.backup_bin is not None:
            defaults["backup_bin"] = os.path.abspath(args.backup_bin)
        if args.backup_workdir is not None:
            defaults["backup_workdir"] = os.path.abspath(args.backup_workdir)
        saved = save_config(cfg, getattr(args, "config", None), defaults=defaults)
        header("CONFIG", f"saved defaults to {saved}")
        for k, v in sorted(defaults.items()):
            kv(k, v)
        return

    if cmd == "remove":
        if args.name not in cfg:
            warn(f"'{args.name}' is not in {path}")
            sys.exit(1)
        del cfg[args.name]
        save_config(cfg, getattr(args, "config", None))
        info(f"removed '{args.name}' from {path}")
        return


def do_topology(args):
    """Show the current replication topology across one or more clusters.

    Queries each cluster's own GetReplicateConfiguration (its self-view of the
    topology), prints each cluster's role (PRIMARY / STANDBY / INDEPENDENT) as
    derived from that view, the union of edges, and a consistency check. Two
    clusters disagreeing usually means a residual edge from an interrupted
    force-promote/teardown — the cluster still pointing at a now-independent
    peer is stuck retrying it.

    Read-only; safe to run any time.
    """
    # Each --cluster / --clusters entry is a config NAME or an inline NAME=URI.
    config = load_config(getattr(args, "config", None))
    cluster_specs = list(args.cluster or [])
    if args.clusters:
        # --clusters is nargs="+", so it arrives as a list of tokens; tolerate
        # commas AND/OR spaces between names ('a,b,c', 'a, b, c', 'a b c').
        raw = args.clusters if isinstance(args.clusters, list) else [args.clusters]
        for tok in raw:
            cluster_specs += [c.strip() for c in tok.replace(",", " ").split() if c.strip()]
    if not cluster_specs:
        # No clusters given → default to every cluster in the config file.
        cluster_specs = sorted(config)
        if not cluster_specs:
            print(f"  {_red('✗')} no clusters given and the config file is empty — "
                  f"use --cluster NAME / --clusters n1,n2,n3, or "
                  f"`ternctl config add`", file=sys.stderr)
            sys.exit(2)
    resolved = [resolve_cluster("query", c, config, pchannel_num=getattr(args, "pchannel_num", None),
                                token=getattr(args, "token", None)) for c in cluster_specs]
    specs = [(cl.cluster_id, cl.dial_addr) for cl in resolved]

    header("TOPOLOGY",
           "querying: " + ", ".join(_cyan(c) for c, _ in specs))

    # Guard: each --cluster must point at a DISTINCT address. Reusing one
    # address (a common mistake — three identical host:port, e.g. forgetting
    # that local port-forwards use different ports) silently queries the same
    # cluster N times and produces a misleading "all agree" result.
    seen = {}
    for cid, addr in specs:
        if addr in seen:
            warn(f"--cluster {_cyan(cid)} and {_cyan(seen[addr])} both point at "
                 f"{addr} — you're querying the SAME cluster twice. Each cluster "
                 f"needs its own address (local port-forwards use different "
                 f"ports, e.g. 19530/19531/19532). Results below are unreliable.")
        seen[addr] = cid

    # Short-timeout + retry: GetReplicateConfiguration hangs to the client
    # deadline on INDEPENDENT clusters, so a single long-timeout call randomly
    # looks "unreachable". See replication.call_with_retry / milvus#50344.
    rpc_timeout = getattr(args, "rpc_timeout", None) or RPC_TIMEOUT
    rpc_tries = getattr(args, "rpc_retries", None) or RPC_RETRIES

    # Query all clusters IN PARALLEL (each may take up to tries×timeout when
    # the hang bites), and print one progress line per cluster as it answers —
    # so a slow run shows liveness instead of a silent stall.
    import concurrent.futures

    def _query_one(cl):
        # Fresh channel PER attempt: a call that deadlines out poisons its
        # HTTP/2 connection, so reusing one channel makes every retry fail too.
        def _call(timeout):
            stub, ch = cl.stub()
            try:
                return stub.GetReplicateConfiguration(
                    milvus_pb2.GetReplicateConfigurationRequest(),
                    metadata=auth_metadata(cl.token), timeout=timeout)
            finally:
                ch.close()
        t0 = time.time()
        try:
            resp = call_with_retry(_call, tries=rpc_tries, timeout=rpc_timeout)
            cfg = resp.configuration
            view = {
                "clusters": [x.cluster_id for x in cfg.clusters],
                "edges": [(t.source_cluster_id, t.target_cluster_id)
                          for t in cfg.cross_cluster_topology],
                "error": None,
            }
        except grpc.RpcError as e:
            view = {"clusters": [], "edges": [],
                    "error": e.code().name if hasattr(e, "code") else str(e)[:40]}
        return cl.cluster_id, view, time.time() - t0

    views = {}  # cid -> {"edges": [(s,t)], "clusters": [...], "error": str|None}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(resolved)) as ex:
        futs = [ex.submit(_query_one, cl) for cl in resolved]
        for fut in concurrent.futures.as_completed(futs):
            cid, view, dt = fut.result()
            views[cid] = view
            state = "ok" if view["error"] is None else view["error"]
            print(_dim(f"  · {cid} answered in {dt:.1f}s ({state})"), flush=True)
    print()

    # ── Forest render ──────────────────────────────────────────────────
    # ONLY edges confirmed by both endpoints build the tree and decide roles.
    # A claim some reachable endpoint denies is a RESIDUAL (e.g. the stale
    # edge a live old-primary keeps after a force-promote): it must not paint
    # the denier with a role it doesn't have — it renders as a dashed warning
    # under the claimer, with the repair command.
    all_edges = {}
    for cid, v in views.items():
        for e in v["edges"]:
            all_edges.setdefault(e, set()).add(cid)
    queried = [cid for cid, _ in specs]
    reachable = {cid for cid in queried if not views[cid]["error"]}

    confirmed, residual = {}, {}
    for e, reporters in all_edges.items():
        deniers = [c for c in e if c in reachable and c not in reporters]
        (residual if deniers else confirmed)[e] = reporters

    children, parents = {}, {}
    for (s_, t_) in confirmed:
        children.setdefault(s_, []).append(t_)
        parents.setdefault(t_, []).append(s_)

    def self_role(cid):
        if cid in children:
            return "PRIMARY"
        if cid in parents:
            return "STANDBY"
        return "INDEPENDENT"

    def residual_lines(cid, prefix):
        """Dashed warning lines for every residual claim this cluster makes."""
        for (s_, t_), reporters in sorted(residual.items()):
            if cid not in reporters:
                continue
            if cid == s_:
                deny = f"{t_} reports {self_role(t_)}"
                print(f"  {prefix}└╌⚠ {_yel(t_):14} "
                      f"{_yel('residual claim →')} {_dim('(unacknowledged; ' + deny + ')')}")
                print(f"  {prefix}    {_dim('repair: ternctl detach --downstream ' + t_ + ' --upstream ' + s_)}")
            else:
                print(f"  {prefix}└╌⚠ {_yel(s_):14} "
                      f"{_yel('residual claim ←')} {_dim('(this cluster believes it is a standby of ' + s_ + ', which does not acknowledge)')}")

    seen = set()

    def render(cid, prefix):
        kids = sorted(children.get(cid, []))
        for i, kid in enumerate(kids):
            last = (i == len(kids) - 1)
            conn = "└── " if last else "├── "
            if kid in seen:
                print(f"  {prefix}{conn}{_red(kid + '  ↻ CYCLE')}")
                continue
            seen.add(kid)
            note = "" if kid in queried else "  " + _dim("(not queried)")
            print(f"  {prefix}{conn}{kid:14} {_yel('STANDBY')}{note}")
            render(kid, prefix + ("    " if last else "│   "))
        residual_lines(cid, prefix)

    roots = sorted([c for c in set(queried) | set(children)
                    if c not in parents and (c in children or c in reachable)])
    first = True
    for r in roots:
        if not first:
            print()
        first = False
        seen.add(r)
        kids = children.get(r, [])
        role = _green("PRIMARY") if kids else _dim("INDEPENDENT")
        extra = "" if kids else "  " + _dim("(no replication edges)")
        note = "" if r in queried else "  " + _dim("(not queried)")
        print(f"  {_green('●')} {r:14} {role}{extra}{note}")
        render(r, "")
    for cid in queried:
        if views[cid]["error"] and cid not in parents and cid not in children:
            print()
            print(f"  {_red('○')} {cid:14} {_red('unreachable')} "
                  f"{_dim('(' + views[cid]['error'] + ')')}")
    print()

    # Consistency summary, aligned with the residual classification above.
    if not reachable:
        warn("no reachable clusters")
        return
    if residual:
        lines = []
        for (s_, t_), reporters in sorted(residual.items()):
            deniers = [c for c in (s_, t_) if c in reachable and c not in reporters]
            lines.append(f"{s_}→{t_} (claimed by {', '.join(sorted(reporters))}; "
                         f"unacknowledged by {', '.join(deniers)})")
        warn(f"{len(residual)} RESIDUAL edge(s) — typically left on a live "
             f"old primary by a force-promote, or by an interrupted topology "
             f"change: " + "; ".join(lines))
    else:
        kv("consistency",
           "consistent — every edge is confirmed by both of its endpoints", _green)

def do_attach(args, upstream, downstream):
    """Register the upstream→downstream replication edge WITHOUT seeding data
    (the inverse of detach). MERGE is the only default semantics: the edge is
    added into the source's current topology, existing edges untouched — for
    an empty/fresh pair this degenerates to the plain single-edge config.
    --replace is the surgery channel: send exactly this single-edge config,
    removals included (poisoned/divergent state repair)."""
    source, target = upstream, downstream
    if getattr(args, "replace", False):
        config = build_replicate_config(upstream, downstream,
                                        source=source, target=target)
    else:
        config = merged_replicate_config(source, target)
    # Order for "both": SOURCE first, TARGET second. Per the CDC design, only
    # the primary actually executes the change (broadcasts the
    # AlterReplicateConfigMessage into its WAL); the call to a standby is a
    # convergence BARRIER — it blocks until the config has arrived via CDC and
    # matches. (A still-INDEPENDENT target accepts the config directly —
    # bootstrap path — so this order is safe there too.)
    targets = {"upstream": [upstream], "downstream": [downstream],
               "both": [source, target]}[args.apply_to]
    # Divergence guard: the merged config is built from the SOURCE's view, so
    # it can never drop a source-side edge — but a receiving cluster whose own
    # view has DIVERGED (residual edges from interrupted operations) would
    # still lose them on apply. That repair must be explicit: --replace.
    if not getattr(args, "replace", False):
        new_edges = {(t.source_cluster_id, t.target_cluster_id)
                     for t in config.cross_cluster_topology}
        for cluster in targets:
            try:
                cur = get_replicate_view(cluster,
                                         getattr(args, "rpc_retries", None),
                                         getattr(args, "rpc_timeout", None))
            except RuntimeError:
                continue  # no readable view → nothing to lose
            removed = {(t.source_cluster_id, t.target_cluster_id)
                       for t in cur.cross_cluster_topology} - new_edges
            if removed:
                lost = ", ".join(f"{s_}→{t_}" for s_, t_ in sorted(removed))
                raise RuntimeError(
                    f"{cluster.cluster_id}'s own view holds edge(s) {lost} that "
                    f"this attach would implicitly remove — its state has "
                    f"diverged. Inspect with `ternctl topology`; if tearing "
                    f"those down is intended, re-run with --replace.")
    for cluster in targets:
        apply_replicate_config(cluster, config)

def discover_upstream(args, downstream, config):
    """A standby has exactly ONE incoming edge — read it from the downstream's
    own replicate config, so `detach --downstream X` needs no --upstream."""
    cfg = get_replicate_view(downstream,
                             getattr(args, "rpc_retries", None),
                             getattr(args, "rpc_timeout", None))
    sources = sorted({t.source_cluster_id for t in cfg.cross_cluster_topology
                      if t.target_cluster_id == downstream.cluster_id})
    if not sources:
        raise RuntimeError(
            f"{downstream.cluster_id} has no incoming replication edge — "
            f"nothing to detach (run `ternctl topology` to see the picture)")
    if len(sources) > 1:
        raise RuntimeError(
            f"{downstream.cluster_id} reports multiple upstreams "
            f"({', '.join(sources)}) — pass --upstream explicitly")
    cid = sources[0]
    if cid not in config:
        raise RuntimeError(
            f"discovered upstream '{cid}' is not in the config file — "
            f"add it (ternctl config add {cid} --uri ...) or pass --upstream inline")
    info(f"upstream auto-discovered from {downstream.cluster_id}'s replicate "
         f"config: {cid}")
    return resolve_cluster("upstream", cid, config, token=getattr(args, "token", None),
                           pchannel_num=getattr(args, "pchannel_num", None))


def do_detach(args, upstream, downstream):
    """Remove ONE replication edge (upstream→downstream), leaving the
    upstream's other downstream edges intact.

    Mechanism (verified end-to-end against milvus v2.6.18):
    - Apply the upstream's current topology MINUS this edge on the upstream
      (full-state replacement API — see merged_replicate_config_minus). With
      no edges left this is the independent config (`clusters=[primary]`,
      `topology=[]`), which milvus accepts: a primary may clear its outbound
      edge by becoming independent.
    - The change BROADCASTS along the existing streams; the removed secondary
      automatically transitions to independent primary — no second call needed.
    - Calling `force_promote=True` on the old secondary AFTER step 1 is
      rejected with "current cluster is primary" — because by then it
      already IS independent primary.

    Note: `clusters=[A,B], topology=[]` (keeping both cluster defs) is
    rejected by the validator ("primary count is not 1"). The independent
    config (single cluster + no topology) is the only shape that works.

    Note: do NOT use this as a "pause" — the source's
    WAL retention window keeps ticking after the edge is removed, so
    re-creating the edge later may silently lose data if retention has
    expired.
    """
    header("DETACH",
           f"primary:   {_cyan(upstream.cluster_id)}\n"
           f"secondary: {_cyan(downstream.cluster_id)}")
    # Remove ONLY the named edge. UpdateReplicateConfiguration is full-state
    # replacement, so the old independent-config approach wiped the primary's
    # ENTIRE topology — with multiple downstreams (a→b plus a→c), breaking
    # a→b silently destroyed a→c as well. The minus-config keeps other edges.
    minus = merged_replicate_config_minus(upstream, downstream.cluster_id)
    remaining = [(t.source_cluster_id, t.target_cluster_id)
                 for t in minus.cross_cluster_topology]
    # Consequence preview BEFORE the prompt — detach is destructive (a working
    # edge is torn down) and cannot be made safe-by-construction, so it earns a
    # confirmation. Show exactly what changes, including the LAST-edge cascade.
    print(f"  this removes edge {_yel(upstream.cluster_id + '→' + downstream.cluster_id)}:")
    print(f"    · {downstream.cluster_id} becomes independent")
    if not remaining:
        print(f"    · this is {upstream.cluster_id}'s LAST edge — "
              f"{upstream.cluster_id} becomes independent too")
    else:
        print(f"    · untouched: " + ", ".join(f"{s_}→{t_}" for s_, t_ in remaining))
    warn("NOT a pause — the source's WAL retention keeps ticking; re-attaching "
         "later may silently lose data. To resume redundancy, use rebuild.")
    if not getattr(args, "yes", False):
        ans = input("  proceed with detach? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            info("aborted")
            sys.exit(1)
    step("1/1", f"apply topology minus {upstream.cluster_id}→{downstream.cluster_id} "
                f"on {upstream.cluster_id}")
    apply_replicate_config(upstream, minus, force_promote=False, _quiet=True); done()
    header("DONE")
    info(f"edge {upstream.cluster_id} → {downstream.cluster_id} removed; "
         f"{downstream.cluster_id} auto-transitions to independent (broadcast)")
    if remaining:
        info("untouched edges: " + ", ".join(f"{s}→{t}" for s, t in remaining))
    else:
        info(f"that was the last edge — {upstream.cluster_id} is now independent too")


