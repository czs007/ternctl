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
    once = getattr(args, "once", False)
    if once:
        retries = 1  # single snapshot — no internal wait loop
    header("VERIFY",
           f"compare row counts: {_cyan(upstream.cluster_id)} (source) vs "
           f"{_cyan(downstream.cluster_id)} (target)"
           + ("\nsnapshot mode (--once): no retry; re-run to watch convergence" if once else ""))
    names = args.collections.split(",") if args.collections else list_collections(upstream)
    names = [n for n in names if n]
    if not names:
        warn("no collections found on the primary; nothing to verify")
        return True
    last_up, last_down = {}, {}
    for attempt in range(1, retries + 1):
        last_up = row_counts(upstream, names)
        last_down = row_counts(downstream, names)
        if all(last_up.get(n) == last_down.get(n) for n in names):
            break
        if attempt < retries:
            info(f"counts differ (attempt {attempt}/{retries}); waiting {interval}s for replication...")
            time.sleep(interval)
    ok = True
    behind = []  # (collection, source - target) for numeric diffs
    head = f"{'collection':32}  {'source':>10}  {'target':>10}   result"
    print("  " + _dim(head))
    print("  " + _dim("─" * 64))
    for n in names:
        up_c, down_c = last_up.get(n), last_down.get(n)
        match = up_c == down_c
        ok = ok and match
        mark = _green("✓ MATCH") if match else _yel("… behind") if once else _red("✗ DIFF")
        print(f"  {n:32}  {str(up_c):>10}  {str(down_c):>10}   {mark}")
        if not match and isinstance(up_c, int) and isinstance(down_c, int):
            behind.append((n, up_c - down_c))
    print()
    if ok:
        kv("result", "OK — counts match", _green)
    elif once:
        # snapshot mode: report the gap, don't declare failure
        gap = ", ".join(f"{n} {d:+d}" for n, d in behind) if behind else "see table"
        kv("snapshot", f"target is behind ({gap}). If the source is being "
           f"written, re-run to watch it catch up; a persistent gap with no "
           f"writes means a real problem.", _yel)
    else:
        kv("result", "FAILED — row counts diverged after retries", _red)
    return ok


# --------------------------------------------------------------------------- #
# Operations
