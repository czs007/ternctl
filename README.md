# ternctl

**Tern** — Milvus cross-cluster replication & disaster recovery, on the CLI.

`ternctl` drives Milvus 2.6's built-in cross-cluster replication for
active-standby DR: seed a standby, switch over gracefully, force-promote when
the primary is down, salvage in-flight data, and watch replication health —
all without leaving the terminal.

> Naming: **Tern** (the migratory seabird) is the replication/DR capability;
> **ternctl** is the CLI that drives it — same relationship as Kubernetes ↔
> `kubectl`.

## Install

```bash
pip install ternctl              # from PyPI
# or, from a checkout:
pip install -e .
```

Requires **Milvus ≥ 2.6.16** on every cluster (that's where `force_promote`
landed). Depends only on `pymilvus` (ships the gRPC proto) and `pyyaml`.

## Quickstart

Define your clusters once (kubectl-config style), then reference them by name:

```bash
# 1. register clusters (writes ~/.ternctl.yaml)
ternctl config add cluster-a --uri http://127.0.0.1:19530 \
  --inter http://cluster-a-milvus.ns-a.svc.cluster.local:19530 \
  --backup-config /abs/path/backup-a.yaml --kafka-brokers 127.0.0.1:19092
ternctl config add cluster-b --uri http://127.0.0.1:19531 \
  --inter http://cluster-b-milvus.ns-b.svc.cluster.local:19530 \
  --backup-config /abs/path/backup-b.yaml
ternctl config set-defaults --backup-bin /abs/path/milvus-backup \
  --backup-workdir /abs/path/workdir

# 2. seed the standby + start replication
# (backup bin/config/workdir come from the config file — no --backup-* needed)
ternctl rebuild   --upstream cluster-a --downstream cluster-b

# 3. watch replication health
ternctl status    --upstream cluster-a    # downstreams auto-discovered
ternctl topology                          # no args = every configured cluster

# 4. graceful switchover (reverse the direction)
ternctl switchover --target cluster-b    # current primary auto-discovered
```

## Specifying clusters: by name or inline

Every command takes clusters two ways:

| form | example | when |
|---|---|---|
| **reference** | `--upstream cluster-a` | normal use — looked up in `~/.ternctl.yaml` (override the file with the `TERNCTL_CONFIG` env var, e.g. one shell per environment) |
| **inline** | `--upstream cluster-a=http://127.0.0.1:19530` | CI / one-off, no config file needed |

## Commands

| command | what it does |
|---|---|
| `rebuild` | seed the standby from a backup + start replication |
| `switchover` | graceful role flip (RPO=0): `--target` = who should END UP primary; current primary auto-discovered |
| `force-promote` | promote a standby to independent primary when the primary is **down** (bounded RPO = CDC lag); can prefetch a salvage checkpoint |
| `status` | per-pchannel replication progress; omit `--downstream` to auto-discover all downstreams; with CDC metrics, the real e2e lag |
| `topology` | replication forest (PRIMARY roots, standbys nested) + per-edge & overall consistency; no args = every configured cluster |
| `verify` | compare row counts (`--once` for a single snapshot); omit `--downstream` to auto-discover |
| `detach` | remove ONE replication edge; `--downstream` alone auto-discovers its upstream (teardown — not a pause) |
| `attach` | register an edge WITHOUT seeding data (inverse of `detach`; merge semantics — existing edges kept; `--replace` for divergent-state surgery) |
| `backup` | `create` / `list` / `get` / `restore` / `delete` — the whole archive lifecycle (config-driven endpoints) |
| `salvage` | dump the unforwarded WAL tail from Kafka after a force-promote; `--source-cluster NAME` sweeps ALL pchannels in one run |
| `replay` | reconcile a salvage dump into any WRITABLE cluster (new primary, or a scratch cluster for inspection) — fill gaps only; conflicts reported, `--overwrite` to let the dump win |
| `clusters` | list the clusters from `~/.ternctl.yaml` (`--probe` checks gRPC reachability) |
| `config` | manage `~/.ternctl.yaml` (`add` / `list` / `show` / `remove` / `set-defaults`) |
| `repl` | interactive shell — run subcommands without re-typing the launcher |

Run `ternctl <command> --help` for full flags.

## Config file (`~/.ternctl.yaml`)

```yaml
clusters:
  cluster-a:
    uri: http://127.0.0.1:19530
    inter_uri: http://cluster-a-milvus.ns-a.svc.cluster.local:19530
  cluster-b:
    uri: http://127.0.0.1:19531
    inter_uri: http://cluster-b-milvus.ns-b.svc.cluster.local:19530
  cluster-c:
    uri: http://127.0.0.1:19532
    inter_uri: http://cluster-c-milvus.ns-c.svc.cluster.local:19530
    cdc_metrics: http://127.0.0.1:9091   # source CDC pod /metrics, for real lag
```

- **`uri`** — the milvus proxy you dial.
- **`inter_uri`** — the internal DNS the *other* cluster uses to reach this one
  (must be a full `http://...` URI reachable from the peer, not `127.0.0.1`).
- **`cdc_metrics`** — optional; this cluster's CDC pod `/metrics` endpoint, used
  to show real replication lag (see below).

Override the path with the `TERNCTL_CONFIG` env var (e.g. one shell per environment).

## Real replication lag (no Prometheus needed)

The true replication delay is the **CDC end-to-end latency** — measured inside
CDC (only it sees each message's source-produce and target-ack times) and
exported on the source CDC pod's `/metrics`. `ternctl status` reads it straight
from there:

```bash
# explicit:
ternctl status --upstream cluster-a --downstream cluster-b \
  --up-cdc http://127.0.0.1:9091
# or set cdc_metrics in the config and it's automatic:
ternctl status --upstream cluster-a --downstream cluster-b
#   ● cluster-b-rootcoord-dml_0   active lag~21ms avg
```

> `status` without CDC metrics shows replication *progress* (which pchannels are
> active/idle), not delay — a correct lag needs the source's latest timetick,
> which only CDC measures.

## Interactive shell

```bash
ternctl repl
ternctl> status --upstream cluster-a --downstream cluster-b
ternctl> topology --clusters cluster-a,cluster-b
ternctl> exit
```

## Disaster-recovery flow

The full surface, in order:

1. **Baseline** — `rebuild` seeds the standby; `status` / `verify` confirm replication.
2. **Planned failover** — `switchover` reverses the direction gracefully.
3. **Unplanned failover** — `force-promote` promotes the standby when the primary
   is down (bounded RPO). Pass `--checkpoint-source` to snapshot a
   salvage checkpoint *before* the promote, while `GetReplicateInfo` still works.
4. **Rebuild** — bring the old primary back as the new standby.

## License

Apache-2.0.
