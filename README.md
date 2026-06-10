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
  --inter http://cluster-a-milvus.ns-a.svc.cluster.local:19530
ternctl config add cluster-b --uri http://127.0.0.1:19531 \
  --inter http://cluster-b-milvus.ns-b.svc.cluster.local:19530

# 2. seed the standby + start replication
ternctl rebuild   --upstream cluster-a --downstream cluster-b

# 3. watch replication health
ternctl status    --upstream cluster-a --downstream cluster-b
ternctl topology  --clusters cluster-a,cluster-b

# 4. graceful switchover (reverse the direction)
ternctl switchover --upstream cluster-a --downstream cluster-b
```

## Specifying clusters: by name or inline

Every command takes clusters two ways:

| form | example | when |
|---|---|---|
| **reference** | `--upstream cluster-a` | normal use — looked up in `~/.ternctl.yaml` |
| **inline** | `--upstream cluster-a=http://127.0.0.1:19530` | CI / one-off, no config file needed |

## Commands

| command | what it does |
|---|---|
| `rebuild` | seed the standby from a backup + start replication |
| `switchover` | gracefully reverse the topology (planned failover) |
| `force-promote` | promote a standby to standalone primary when the primary is **down** (bounded RPO = CDC lag); can prefetch a salvage checkpoint |
| `status` | per-pchannel replication progress; with CDC metrics, the real e2e lag |
| `topology` | show the replication topology across clusters + a consistency check |
| `verify` | compare row counts across clusters (`--once` for a single snapshot) |
| `break-topology` | delete a replication edge (teardown — not a pause) |
| `replicate-config` | low-level: apply a replicate configuration directly |
| `config` | manage `~/.ternctl.yaml` (`add` / `list` / `show` / `remove`) |
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

Override the path with `--config PATH` or `TERNCTL_CONFIG`.

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
   is down (bounded RPO). Pass `--salvage-source-cluster-id` to snapshot a
   salvage checkpoint *before* the promote, while `GetReplicateInfo` still works.
4. **Rebuild** — bring the old primary back as the new standby.

## License

Apache-2.0.
