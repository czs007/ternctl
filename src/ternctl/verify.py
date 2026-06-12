"""Row-count verification across clusters."""
import time

from pymilvus import MilvusClient

from .output import header, kv, info, warn, _green, _red, _yel, _cyan, _dim


def _client(cluster):
    return MilvusClient(uri=cluster.uri, token=cluster.token)


def list_collections(cluster):
    c = _client(cluster)
    try:
        return c.list_collections()
    finally:
        c.close()


def row_counts(cluster, names):
    c = _client(cluster)
    counts = {}
    try:
        for name in names:
            try:
                counts[name] = c.get_collection_stats(name)["row_count"]
            except Exception as exc:
                counts[name] = f"ERR: {exc}"
    finally:
        c.close()
    return counts


def verify(args, upstream, downstream, retries=6, interval=5):
    return verify_many(args, upstream, [downstream], retries, interval)


def verify_many(args, upstream, downstreams, retries=6, interval=5):
    """ONE table for any number of downstreams: a column per target, a row per
    collection. The retry loop waits until EVERY pair converges (or budget
    runs out); --once takes a single snapshot and reports gaps without
    declaring failure."""
    once = getattr(args, "once", False)
    if once:
        retries = 1
    tids = [d.cluster_id for d in downstreams]
    header("VERIFY",
           f"row counts: {_cyan(upstream.cluster_id)} (source) vs "
           + ", ".join(_cyan(t) for t in tids)
           + ("\nsnapshot mode (--once): no retry; re-run to watch convergence" if once else ""))
    names = args.collections.split(",") if args.collections else list_collections(upstream)
    names = [n for n in names if n]
    if not names:
        warn("no collections found on the primary; nothing to verify")
        return True

    src, per_t = {}, {}
    for attempt in range(1, retries + 1):
        src = row_counts(upstream, names)
        per_t = {d.cluster_id: row_counts(d, names) for d in downstreams}
        if all(src.get(n) == per_t[t].get(n) for n in names for t in tids):
            break
        if attempt < retries:
            info(f"counts differ (attempt {attempt}/{retries}); waiting {interval}s for replication...")
            time.sleep(interval)

    cw = max([len("collection")] + [len(n) for n in names]) + 2
    tw = [max(len(t), 12) for t in tids]
    head = f"{'collection':<{cw}}{'source':>10}  " + "  ".join(f"{t:>{w}}" for t, w in zip(tids, tw))
    print("  " + _dim(head))
    print("  " + _dim("─" * len(head)))
    ok = True
    gaps = []   # (collection, target, source-target) numeric gaps
    for n in names:
        cells = []
        for t, w in zip(tids, tw):
            sc, tc = src.get(n), per_t[t].get(n)
            match = sc == tc
            ok = ok and match
            mark = _green("✓") if match else (_yel("…") if once else _red("✗"))
            cells.append(f"{str(tc):>{w - 2}} {mark}")
            if not match and isinstance(sc, int) and isinstance(tc, int):
                gaps.append((n, t, sc - tc))
        print(f"  {n:<{cw}}{str(src.get(n)):>10}  " + "  ".join(cells))
    print()
    if ok:
        kv("result", f"OK — all {len(tids)} downstream(s) match", _green)
    elif once:
        gap = ", ".join(f"{n}@{t} {d:+d}" for n, t, d in gaps) if gaps else "see table"
        kv("snapshot", f"behind: {gap}. If the source is being written, re-run "
           f"to watch it catch up; a persistent gap with no writes means a "
           f"real problem.", _yel)
    else:
        bad = sorted({t for _, t, _ in gaps}) or tids
        kv("result", f"FAILED — diverged after retries: {', '.join(bad)}", _red)
    return ok



# --------------------------------------------------------------------------- #
# Operations
