"""Replay — reconcile a salvage dump into the NEW primary, safely.

The dump (ternctl salvage) is a SUPERSET of the lost data: the checkpoint
records last-CONFIRMED, not last-applied, so an overlap region of already-
replicated rows precedes the genuinely stranded tail. Worse, the new primary
may have taken application writes AFTER the promote — the old primary's
stranded values are the PAST; the new primary's rows are the PRESENT.

Default semantics therefore: FILL GAPS ONLY.
  - fold the dump per primary key (last op by time_tick wins),
  - look each PK up on the target,
  - PK absent  + final op upsert → recover (write it),
  - PK absent  + final op delete → no-op,
  - PK PRESENT + anything       → CONFLICT: skipped and reported, never
    guessed. (--overwrite applies the dump's value/delete anyway — only for
    operators who KNOW no post-failover write touched these keys.)

The conflict report includes the harmless overlap region (rows identical on
both sides) — the tool cannot distinguish those from true post-failover
conflicts without value comparison; investigate only keys your application
may have written after the failover.
"""
import glob
import json
import os
import sys
import time

from pymilvus import MilvusClient
from pymilvus.grpc_gen import msg_pb2

from .output import header, step, done, kv, info, warn, _green, _red, _yel, _cyan, _bold, _dim

_INSERT, _DELETE = 2, 3
_TXN_TYPES = {900, 901, 902}
_BATCH = 1000          # rows per upsert / pks per existence query


# ── decode ─────────────────────────────────────────────────────────────────
def _scalar_column(fd):
    sc = fd.scalars
    for k in ("long_data", "int_data", "string_data", "bool_data",
              "float_data", "double_data"):
        if sc.HasField(k):
            return list(getattr(sc, k).data)
    if sc.HasField("json_data"):
        return [json.loads(b) if b else None for b in sc.json_data.data]
    raise RuntimeError(
        f"field '{fd.field_name}': unsupported scalar payload — replay v1 "
        f"covers long/int/string/bool/float/double/json")


def _vector_column(fd, n):
    v = fd.vectors
    if v.HasField("float_vector"):
        flat, dim = list(v.float_vector.data), v.dim
        return [flat[i * dim:(i + 1) * dim] for i in range(n)]
    raise RuntimeError(
        f"field '{fd.field_name}': unsupported vector kind — replay v1 "
        f"covers float_vector only")


def _insert_rows(req):
    """msg InsertRequest (columnar) → list of row dicts."""
    n = req.num_rows
    cols = {}
    for fd in req.fields_data:
        cols[fd.field_name] = (_vector_column(fd, n) if fd.HasField("vectors")
                               else _scalar_column(fd))
    return [{name: col[i] for name, col in cols.items()} for i in range(n)]


def _delete_pks(req):
    pk = req.primary_keys
    if pk.HasField("int_id"):
        return list(pk.int_id.data)
    if pk.HasField("str_id"):
        return list(pk.str_id.data)
    return []


# ── load & fold ────────────────────────────────────────────────────────────
def load_and_fold(from_dir, only_collections=None):
    """Parse every *.jsonl in from_dir; return
    ({collection: {pk: (tick, 'upsert', row) | (tick, 'delete', None)}}, stats)."""
    files = sorted(glob.glob(os.path.join(from_dir, "*.jsonl")))
    if not files:
        raise RuntimeError(f"no .jsonl files in {from_dir}")
    import base64
    folded, skipped = {}, {}
    msgs = {"Insert": 0, "Delete": 0}
    for path in files:
        for line in open(path):
            r = json.loads(line)
            tid = r.get("type_id")
            if tid in _TXN_TYPES:
                raise RuntimeError(
                    "transaction markers (BeginTxn/CommitTxn/AbortTxn) present "
                    "in the dump — transactional groups must be replayed "
                    "atomically, which replay v1 does not do. Refusing rather "
                    "than corrupting; replay this dump with your own tooling.")
            if tid not in (_INSERT, _DELETE):
                skipped[r.get("type", "?")] = skipped.get(r.get("type", "?"), 0) + 1
                continue
            raw = base64.b64decode(r["body_b64"])
            tick = r["time_tick"]
            if tid == _INSERT:
                req = msg_pb2.InsertRequest()
                req.ParseFromString(raw)
                coll = req.collection_name
                if only_collections and coll not in only_collections:
                    continue
                msgs["Insert"] += 1
                dst = folded.setdefault(coll, {})
                pkf = None  # resolved lazily by caller via schema; fold by row later
                for row in _insert_rows(req):
                    dst.setdefault("_rows", []).append((tick, row))
            else:
                req = msg_pb2.DeleteRequest()
                req.ParseFromString(raw)
                coll = req.collection_name
                if only_collections and coll not in only_collections:
                    continue
                msgs["Delete"] += 1
                dst = folded.setdefault(coll, {})
                for pk in _delete_pks(req):
                    dst.setdefault("_dels", []).append((tick, pk))
    return folded, msgs, skipped


def fold_by_pk(raw_ops, pk_field):
    """(tick, row)/(tick, pk) streams → {pk: (tick, op, row|None)}, last tick wins."""
    final = {}
    for tick, row in raw_ops.get("_rows", []):
        pk = row[pk_field]
        if pk not in final or tick >= final[pk][0]:
            final[pk] = (tick, "upsert", row)
    for tick, pk in raw_ops.get("_dels", []):
        if pk not in final or tick >= final[pk][0]:
            final[pk] = (tick, "delete", None)
    return final


# ── reconcile & apply ──────────────────────────────────────────────────────
def _pk_field(client, coll):
    desc = client.describe_collection(coll)
    for f in desc.get("fields", []):
        if f.get("is_primary") or f.get("is_primary_key"):
            return f["name"], f.get("type")
    raise RuntimeError(f"{coll}: cannot identify the primary key field")


def _existing_pks(client, coll, pk_field, pks):
    out = set()
    str_pk = bool(pks) and isinstance(next(iter(pks)), str)
    pl = sorted(pks)
    for i in range(0, len(pl), _BATCH):
        chunk = pl[i:i + _BATCH]
        lit = ", ".join(f'"{p}"' for p in chunk) if str_pk else ", ".join(map(str, chunk))
        for hit in client.query(coll, filter=f"{pk_field} in [{lit}]",
                                output_fields=[pk_field]):
            out.add(hit[pk_field])
    return out


def do_replay(args, into):
    header("REPLAY",
           f"salvage dump {_cyan(args.from_dir)} → {_cyan(into.cluster_id)} ({into.uri})\n"
           f"default semantics: FILL GAPS ONLY — existing keys on the target "
           f"are never touched" + (f"\n{_red('--overwrite: the dump WINS over the target')}"
                                   if getattr(args, "overwrite", False) else ""))
    only = set(args.collections.split(",")) if getattr(args, "collections", None) else None
    folded, msgs, skipped = load_and_fold(args.from_dir, only)
    if skipped:
        info("skipped WAL machinery: "
             + ", ".join(f"{v} {k}" for k, v in sorted(skipped.items())))
    if not folded:
        warn("no Insert/Delete messages in the dump — nothing to replay")
        return True

    client = MilvusClient(uri=into.uri, token=into.token)
    plan = {}     # coll -> dict(recover=[(pk,row)], drop_dels=n, conflicts=[(pk,op,tick)])
    for coll, raw_ops in folded.items():
        pk_field, _ = _pk_field(client, coll)
        final = fold_by_pk(raw_ops, pk_field)
        existing = _existing_pks(client, coll, pk_field, set(final))
        recover, noop_del, conflicts = [], 0, []
        for pk, (tick, op, row) in final.items():
            if pk in existing:
                conflicts.append((pk, op, tick))
            elif op == "upsert":
                recover.append((pk, row))
            else:
                noop_del += 1
        plan[coll] = {"pk_field": pk_field, "recover": recover,
                      "noop_del": noop_del, "conflicts": conflicts}

    # ── report ──
    total_recover = total_conflict = 0
    for coll, p in plan.items():
        total_recover += len(p["recover"])
        total_conflict += len(p["conflicts"])
        print(f"  {_cyan(coll)}  ({msgs['Insert']} inserts / {msgs['Delete']} deletes in dump)")
        print(f"    recover (pk absent on target):   {_green(str(len(p['recover'])))} rows")
        print(f"    no-op deletes (pk already gone): {p['noop_del']}")
        print(f"    conflicts (pk EXISTS on target): {_yel(str(len(p['conflicts'])))}"
              f"  {_dim('— includes the harmless already-replicated overlap;')}")
        print(f"      {_dim('investigate only keys your app may have written post-failover')}")
    conflict_path = os.path.join(args.from_dir, "conflicts.jsonl")
    if total_conflict:
        with open(conflict_path, "w") as fh:
            for coll, p in plan.items():
                for pk, op, tick in sorted(p["conflicts"]):
                    fh.write(json.dumps({"collection": coll, "pk": pk,
                                         "dump_op": op, "time_tick": tick}) + "\n")
        kv("conflict report", conflict_path, _yel)

    if getattr(args, "dry_run", False):
        kv("dry-run", "no writes performed", _green)
        return True
    todo = total_conflict if getattr(args, "overwrite", False) else 0
    if total_recover + todo == 0:
        info("nothing to write — target already complete")
        return True

    if not getattr(args, "yes", False):
        extra = (f" and OVERWRITE {total_conflict} existing keys"
                 if getattr(args, "overwrite", False) and total_conflict else "")
        ans = input(f"write {total_recover} recovered rows into "
                    f"{into.cluster_id}{extra}? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            info("aborted")
            sys.exit(1)

    for coll, p in plan.items():
        rows = [r for _, r in p["recover"]]
        if getattr(args, "overwrite", False):
            final = fold_by_pk(folded[coll], p["pk_field"])
            del_pks = []
            for pk, op, _t in p["conflicts"]:
                _, fop, frow = final[pk]
                if fop == "upsert":
                    rows.append(frow)
                else:
                    del_pks.append(pk)
            if del_pks:
                step("del", f"{coll}: apply {len(del_pks)} dump deletes (overwrite)")
                client.delete(coll, ids=del_pks)
                done()
        if rows:
            step("ins", f"{coll}: upsert {len(rows)} rows")
            for i in range(0, len(rows), _BATCH):
                client.upsert(coll, rows[i:i + _BATCH])
            done()
    header("DONE")
    kv("recovered", f"{total_recover} rows", _green)
    if total_conflict and not getattr(args, "overwrite", False):
        kv("left for the business to judge", f"{total_conflict} keys → {conflict_path}", _yel)
    info(f"verify with:  ternctl verify --upstream {into.cluster_id}   (or smoke_query)")
    return True
