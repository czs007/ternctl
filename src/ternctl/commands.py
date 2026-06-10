"""Subcommand implementations: do_rebuild / switchover / force_promote / ..."""
import datetime
import json
import os
import sys
import time

import grpc
from pymilvus.grpc_gen import milvus_pb2

from .output import (header, step, done, info, warn, kv, _green, _red, _yel, _cyan, _dim, _bold)
from .cluster import pchannels_of, auth_metadata
from .config import load_config, save_config, resolve_cluster, config_path
from .replication import (build_replicate_config, apply_replicate_config,
                          independent_replicate_config, get_replicate_checkpoints, fetch_cdc_latency,
                          prefetch_salvage_checkpoints, call_with_retry,
                          RPC_RETRIES, RPC_TIMEOUT)
from .backup import backup_create, restore_secondary, restore_backup
from .verify import verify


# --------------------------------------------------------------------------- #
def do_rebuild(args, upstream, downstream):
    header("REBUILD",
           f"source: {_cyan(upstream.cluster_id)} ({upstream.uri})\n"
           f"target: {_cyan(downstream.cluster_id)} ({downstream.uri})")
    step("1/4", "snapshot primary"); backup_create(args); done()
    step("2/4", "register topology on target")
    up2down = build_replicate_config(upstream, downstream, source=upstream, target=downstream)
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
         f"{_bold('ternctl restore')}, or rebuild a fresh standby from the current primary.")


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


def do_switchover(args, upstream, downstream):
    header("SWITCHOVER",
           f"current primary: {_cyan(upstream.cluster_id)} ({upstream.uri})\n"
           f"current standby: {_cyan(downstream.cluster_id)} ({downstream.uri})")
    step("1/2", f"apply reversed topology to {upstream.cluster_id}")
    down2up = build_replicate_config(upstream, downstream, source=downstream, target=upstream)
    apply_replicate_config(upstream, down2up, _quiet=True); done()
    step("2/2", f"apply reversed topology to {downstream.cluster_id}")
    apply_replicate_config(downstream, down2up, _quiet=True); done()
    header("DONE")
    kv("new primary", f"{downstream.cluster_id} ({downstream.uri})", _green)
    kv("now standby", f"{upstream.cluster_id} ({upstream.uri})", _dim)
    info("point application writes at the new primary; old primary now receives via CDC")


def do_force_promote(args, target):
    do_prefetch = bool(args.salvage_source_cluster_id)
    subtitle = (
        f"target: {_cyan(target.cluster_id)} ({target.uri})\n"
        f"{_yel('⚠')} RPO bounded by CDC lag at failure time — NOT zero"
    )
    if do_prefetch:
        subtitle += (
            f"\nprefetch salvage checkpoint from source "
            f"{_cyan(args.salvage_source_cluster_id)} before flipping role"
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
        entries = prefetch_salvage_checkpoints(target, args.salvage_source_cluster_id, pchannels)
        ok = sum(1 for e in entries if e.get("status") == "ok")
        salvage_out_path = args.salvage_output or os.path.abspath(
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
                "source_cluster_id": args.salvage_source_cluster_id,
                "pchannel_num": target.pchannel_num,
                "pchannels": entries,
                "_note": (
                    "ReplicateCheckpoint snapshot taken while target was still in "
                    "SECONDARY state, just before the force_promote RPC. Feed this "
                    "file into ternctl salvage --from-checkpoint-file to "
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
        info(f"to recover from {args.salvage_source_cluster_id}'s WAL, feed it into:")
        info(f"  {_bold('ternctl salvage --from-checkpoint-file ' + salvage_out_path + ' --source-pchannel <topic> --kafka-brokers <hosts> --output salvage.jsonl')}")
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
            e = cfg[name]
            extras = []
            if e.get("inter_uri"):   extras.append("inter=" + e["inter_uri"])
            if e.get("cdc_metrics"): extras.append("cdc=" + e["cdc_metrics"])
            if e.get("token"):       extras.append("token=set")
            print(f"  {_cyan(name):18} {_dim('uri=')}{e.get('uri','?')}"
                  + ("  " + _dim(" ".join(extras)) if extras else ""))
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
        cfg[args.name] = entry
        saved = save_config(cfg, getattr(args, "config", None))
        header("CONFIG", f"saved '{_cyan(args.name)}' to {saved}")
        kv("uri", entry["uri"], _green)
        if entry.get("inter_uri"):   kv("inter_uri", entry["inter_uri"])
        if entry.get("cdc_metrics"): kv("cdc_metrics", entry["cdc_metrics"])
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
        cluster_specs += [c.strip() for c in args.clusters.split(",") if c.strip()]
    if not cluster_specs:
        print(f"  {_red('✗')} give clusters via --cluster NAME (repeatable) or "
              f"--clusters n1,n2,n3", file=sys.stderr)
        sys.exit(2)
    resolved = [resolve_cluster("query", c, config, pchannel_num=args.pchannel_num,
                                token=args.token) for c in cluster_specs]
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

    views = {}  # cid -> {"edges": [(s,t)], "clusters": [...], "error": str|None}
    for cl in resolved:
        cid = cl.cluster_id
        # Fresh channel PER attempt: a call that deadlines out poisons its
        # HTTP/2 connection, so reusing one channel makes every retry fail too.
        def _call(timeout, _cl=cl):
            stub, ch = _cl.stub()
            try:
                return stub.GetReplicateConfiguration(
                    milvus_pb2.GetReplicateConfigurationRequest(),
                    metadata=auth_metadata(_cl.token), timeout=timeout)
            finally:
                ch.close()
        try:
            resp = call_with_retry(_call, tries=rpc_tries, timeout=rpc_timeout)
            cfg = resp.configuration
            views[cid] = {
                "clusters": [x.cluster_id for x in cfg.clusters],
                "edges": [(t.source_cluster_id, t.target_cluster_id)
                          for t in cfg.cross_cluster_topology],
                "error": None,
            }
        except grpc.RpcError as e:
            views[cid] = {"clusters": [], "edges": [],
                          "error": e.code().name if hasattr(e, "code") else str(e)[:40]}

    # Per-cluster role, derived from each cluster's OWN view.
    for cid, _ in specs:
        v = views[cid]
        if v["error"]:
            print(f"  {_red('○')} {cid:14} {_red('unreachable')} {_dim('(' + v['error'] + ')')}")
            continue
        outgoing = [t for (s, t) in v["edges"] if s == cid]
        incoming = [s for (s, t) in v["edges"] if t == cid]
        if outgoing:
            role, detail = _green("PRIMARY"), "→ " + ", ".join(outgoing)
        elif incoming:
            role, detail = _yel("STANDBY"), "← " + ", ".join(incoming)
        else:
            role, detail = _dim("INDEPENDENT"), _dim("(no replication edges)")
        print(f"  {_green('●')} {cid:14} {role}  {detail}")

    # Union of edges across all views.
    all_edges = {}
    for cid, v in views.items():
        for e in v["edges"]:
            all_edges.setdefault(e, set()).add(cid)
    if all_edges:
        print()
        print(f"  {_dim('edges (source → target):')}")
        for (s, t), reporters in sorted(all_edges.items()):
            print(f"    {_bold(s + ' → ' + t)}   {_dim('reported by ' + ', '.join(sorted(reporters)))}")

    # Consistency: do reachable clusters agree on the edge set?
    reachable = {cid: frozenset(v["edges"]) for cid, v in views.items() if v["error"] is None}
    print()
    if not reachable:
        warn("no reachable clusters")
    elif len(set(reachable.values())) <= 1:
        kv("consistency", "all reachable clusters agree on the topology", _green)
    else:
        warn("clusters DISAGREE on the topology — likely a residual edge from an "
             "interrupted force-promote/teardown (a cluster still pointing at a "
             "now-independent peer keeps retrying it).")
        for cid, es in reachable.items():
            shown = ", ".join(f"{s}→{t}" for (s, t) in sorted(es)) or "independent"
            print(f"    {cid}: {shown}")


def do_replicate_config(args, upstream, downstream):
    source, target = (upstream, downstream) if args.direction == "up2down" else (downstream, upstream)
    config = build_replicate_config(upstream, downstream, source=source, target=target)
    targets = {"upstream": [upstream], "downstream": [downstream], "both": [downstream, upstream]}[args.target]
    for cluster in targets:
        apply_replicate_config(cluster, config)


def do_break_topology(args, upstream, downstream):
    """Tear down a single-edge replication topology by applying an independent
    config to the PRIMARY side only.

    Mechanism (verified end-to-end against milvus v2.6.18):
    - Apply `clusters=[primary]`, `topology=[]`, `force_promote=False` on the
      primary cluster. Milvus accepts: a primary is allowed to clear its
      outbound edge by becoming independent.
    - The change BROADCASTS to all clusters in the prior topology. The old
      secondary (downstream) automatically transitions to independent primary
      too — no second call needed.
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
    header("BREAK TOPOLOGY",
           f"primary:   {_cyan(upstream.cluster_id)}\n"
           f"secondary: {_cyan(downstream.cluster_id)}\n"
           f"{_yel('⚠')} DELETES the edge — use only for teardown, not as a pause")
    step("1/1", f"apply independent config on primary ({upstream.cluster_id})")
    apply_replicate_config(upstream, independent_replicate_config(upstream),
                           force_promote=False, _quiet=True); done()
    header("DONE")
    info(f"edge {upstream.cluster_id} → {downstream.cluster_id} removed")
    info("both clusters auto-transition to independent primary (broadcast)")


