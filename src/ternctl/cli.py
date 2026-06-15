"""Argument parser, command dispatch, REPL, and the `ternctl` entry point."""
import argparse
import re
import sys
import time

from . import output
from .output import log, info, warn, _red, _cyan, _bold, _dim
from .config import load_config, load_defaults, resolve_cluster
from .verify import verify, verify_many
from .commands import (do_rebuild, do_switchover, do_force_promote, do_status,
                       do_config, do_topology, do_attach, do_detach, discover_upstream,
                       do_backup, do_restore, do_clusters, do_backups,
                       do_backup_get, do_backup_delete, for_each_downstream,
                       discover_downstreams)
from .salvage import do_salvage
from .replay import do_replay


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_common(p, downstream_required=True, upstream_required=True):
    g = p.add_argument_group("clusters (NAME from config file, or NAME=URI inline)")
    g.add_argument("--upstream", required=upstream_required, metavar="NAME[=URI]",
                   help="source cluster: a name from ~/.ternctl.yaml, or inline "
                        "NAME=URI (e.g. cluster-a=http://127.0.0.1:19530)"
                        + ("" if upstream_required else
                           "; omit to auto-discover it from --downstream's own "
                           "replicate config (a standby has exactly one upstream)"))
    g.add_argument("--downstream", required=downstream_required, metavar="NAME[=URI]",
                   help="target cluster (same NAME or NAME=URI form)"
                        + ("" if downstream_required else
                           "; omit to auto-discover ALL downstreams of --upstream "
                           "from its replicate configuration"))
    g.add_argument("--upstream-inter", default=None, metavar="URI",
                   help="override the upstream's inter-cluster URI")
    g.add_argument("--downstream-inter", default=None, metavar="URI",
                   help="override the downstream's inter-cluster URI")


def add_rpc_opts(p):
    g = p.add_argument_group("replication-RPC retry (topology/status)")
    g.add_argument("--rpc-timeout", type=float, default=None, metavar="SECONDS",
                   help="per-attempt timeout for GetReplicateConfiguration / "
                        "GetReplicateInfo (default 3s). These RPCs hang to the "
                        "deadline on INDEPENDENT clusters; a short timeout + retry "
                        "avoids false 'unreachable'. See milvus-io/milvus#50344.")
    g.add_argument("--rpc-retries", type=int, default=None, metavar="N",
                   help="how many short-timeout attempts before declaring a "
                        "cluster unreachable (default 6)")


def add_backup(p):
    # Defaults are None so explicit flags can be told apart from "not given";
    # fill_backup_args() then falls back to the config file (per-cluster
    # backup_config, defaults.backup_bin/backup_workdir) and finally to the
    # historical hardcoded defaults.
    g = p.add_argument_group("milvus-backup (all optional if set in ~/.ternctl.yaml)")
    g.add_argument("--backup-bin", default=None,
                   help="milvus-backup binary (config: defaults.backup_bin)")
    g.add_argument("--backup-workdir", default=None,
                   help="milvus-backup working dir (config: defaults.backup_workdir)")
    g.add_argument("--backup-name", default=None,
                   help="snapshot name in the archive (default: "
                        "ternctl_backup_<timestamp> — unique per run, because a "
                        "create with an existing name fails)")
    g.add_argument("--backup-config", default=None,
                   help="backup config for the SOURCE cluster "
                        "(config: the upstream cluster's backup_config)")
    g.add_argument("--backup-config-secondary", default=None,
                   help="backup config for the TARGET cluster "
                        "(config: the downstream cluster's backup_config)")
    g.add_argument("--no-backup-index-extra", dest="backup_index_extra", action="store_false")
    g.add_argument("--backup-create-extra", nargs=argparse.REMAINDER, default=[])


def fill_backup_args(args, config, up_cid=None, down_cid=None):
    """Resolve the milvus-backup arguments: explicit flag > config file >
    historical default. up_cid/down_cid are config-file cluster names whose
    `backup_config` fields feed --backup-config / --backup-config-secondary."""
    defaults = load_defaults(getattr(args, "config", None))
    if getattr(args, "backup_name", None) is None:
        # Embed the source cluster in the name: backup meta records NO origin
        # cluster, so the name is the only attribution a shared archive gets.
        # milvus-backup only accepts [A-Za-z0-9_] in names — sanitize the id
        # (cluster-b → cluster_b).
        tag = (re.sub(r"[^A-Za-z0-9_]", "_", up_cid) + "_") if up_cid else ""
        args.backup_name = f"ternctl_backup_{tag}{time.strftime('%Y%m%d_%H%M%S')}"
    if getattr(args, "backup_bin", None) is None:
        args.backup_bin = defaults.get("backup_bin") or "./milvus-backup"
    if getattr(args, "backup_workdir", None) is None:
        args.backup_workdir = defaults.get("backup_workdir") or "."
    if getattr(args, "backup_config", None) is None and up_cid:
        args.backup_config = (config.get(up_cid) or {}).get("backup_config")
    if getattr(args, "backup_config", None) is None:
        args.backup_config = "backup.yaml"
    if hasattr(args, "backup_config_secondary"):
        if args.backup_config_secondary is None and down_cid:
            args.backup_config_secondary = (config.get(down_cid) or {}).get("backup_config")
        if args.backup_config_secondary is None:
            args.backup_config_secondary = "backup.yaml"
    return args


def clusters_from_args(args):
    config = load_config(getattr(args, "config", None))
    upstream = resolve_cluster("upstream", args.upstream, config,
                               inter=args.upstream_inter, token=getattr(args, "token", None),
                               pchannel_num=getattr(args, "pchannel_num", None))
    downstream = resolve_cluster("downstream", args.downstream, config,
                                 inter=args.downstream_inter, token=getattr(args, "token", None),
                                 pchannel_num=getattr(args, "pchannel_num", None))
    return upstream, downstream


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ternctl",
        description="Milvus active-standby DR (rebuild / switchover / force-promote).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true",
                        help="stream milvus-backup CLI output to the terminal "
                             "(default: redirect to milvus-backup-cli.log inside backup-workdir)")
    parser.add_argument("--no-color", action="store_true",
                        help="disable ANSI color in output (also auto-disabled when stdout is not a TTY)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_rebuild = sub.add_parser("rebuild", help="seed standby + start replication")
    add_common(p_rebuild)
    add_backup(p_rebuild)
    p_rebuild.add_argument("--verify", action="store_true")
    p_rebuild.add_argument("--collections", default=None)

    p_switch = sub.add_parser("switchover",
                              help="graceful role flip (RPO=0): make --target the primary of its edge")
    g = p_switch.add_argument_group("target (NAME from config file, or NAME=URI inline)")
    g.add_argument("--target", required=True, metavar="NAME[=URI]",
                   help="the standby that should END UP primary; its current "
                        "primary is auto-discovered from its own replicate config")
    add_rpc_opts(p_switch)

    p_force = sub.add_parser("force-promote", help="promote secondary to independent primary (original primary down)")
    g = p_force.add_argument_group("target (NAME from config, or NAME=URI inline)")
    g.add_argument("--target", required=True, metavar="NAME[=URI]",
                   help="the secondary being promoted: a config name, or NAME=URI inline")
    g.add_argument("--target-inter", default=None, metavar="URI",
                   help="override the target's inter-cluster URI")
    p_force.add_argument("--yes", action="store_true", help="skip the RPO confirmation prompt")
    sg = p_force.add_argument_group("salvage checkpoint prefetch (recommended for DR)")
    sg.add_argument("--salvage-from", default=None,
                    help="cluster id of the OLD primary you may want to salvage data from. "
                         "When set, the tool snapshots the live ReplicateCheckpoint for every "
                         "target pchannel BEFORE the force_promote RPC, while GetReplicateInfo "
                         "still works. After force_promote that API is broken on an independent "
                         "primary — see milvus-io/milvus#50344. Without this flag, salvage of "
                         "in-flight messages from the dead primary is not possible.")
    sg.add_argument("--checkpoint-file", default=None,
                    help="output path for the prefetched checkpoint JSON. "
                         "Default: ./salvage_checkpoint_<target>_<unix_ts>.json")
    sg.add_argument("--no-salvage", action="store_true",
                    help="skip the salvage-checkpoint prefetch entirely. Without this "
                         "flag, omitting --salvage-from makes ternctl "
                         "AUTO-DISCOVER the source from the target's own replicate "
                         "configuration (its incoming edge) — the prefetch is read-only "
                         "and skipping it makes the old primary's in-flight data "
                         "unrecoverable, so opting OUT is the explicit action.")

    p_status = sub.add_parser("status", help="dump replication checkpoints")
    add_common(p_status, downstream_required=False, upstream_required=False)
    add_rpc_opts(p_status)
    p_status.add_argument("--upstream-cdc-metrics", "--up-cdc", default=None, metavar="URL",
                          dest="upstream_cdc_metrics",
                          help="override the UPSTREAM (source) cluster's CDC pod metrics "
                               "endpoint, e.g. http://127.0.0.1:9091 (short alias: "
                               "--up-cdc). Taken from the config file's cdc_metrics if "
                               "set there. When available, status also shows the real "
                               "e2e replication lag per pchannel, read straight from "
                               "/metrics (no Prometheus).")

    p_topo = sub.add_parser("topology",
                            help="show the current replication topology across clusters")
    p_topo.add_argument("--cluster", action="append", default=None, metavar="NAME[=URI]",
                        help="a cluster to query: a config NAME or inline NAME=URI "
                             "(repeat for each cluster, or use --clusters)")
    p_topo.add_argument("--clusters", default=None, nargs="+", metavar="N1,N2,N3",
                        help="cluster names from the config file, separated by "
                             "commas and/or spaces (shorthand for repeating "
                             "--cluster). 'a,b,c', 'a, b, c' and 'a b c' all work.")
    add_rpc_opts(p_topo)

    p_verify = sub.add_parser("verify", help="compare row counts")
    add_common(p_verify, downstream_required=False, upstream_required=False)
    add_rpc_opts(p_verify)
    p_verify.add_argument("--collections", default=None)
    p_verify.add_argument("--once", action="store_true",
                          help="single snapshot — skip the internal retry/wait loop. "
                               "Shows both sides' current counts and, if they differ, "
                               "how far behind the target is, without declaring FAILED. "
                               "Re-run it (or `watch`) to watch the standby converge "
                               "while the source is being written.")

    p_attach = sub.add_parser("attach",
                               help="register the upstream→downstream edge WITHOUT seeding "
                                    "data (inverse of detach; use rebuild when the source "
                                    "holds data the target lacks)")
    add_common(p_attach)
    add_rpc_opts(p_attach)
    p_attach.add_argument("--apply-to", choices=["upstream", "downstream", "both"], default="both",
                          help="which cluster(s) to send the RPC to. Only the config's "
                               "PRIMARY actually executes the change (it broadcasts the "
                               "AlterReplicateConfigMessage into its WAL; the config then "
                               "reaches standbys via CDC itself). Sending to a standby is "
                               "a convergence BARRIER: the RPC blocks until the config has "
                               "arrived and matches. 'both' = source first (execute), then "
                               "target (confirm).")
    p_attach.add_argument("--replace", action="store_true",
                          help="surgery: send exactly the single-edge config — edges absent "
                               "from it are TORN DOWN on the receiving cluster (divergent-"
                               "state repair). Default semantics is MERGE: existing edges "
                               "are always kept.")

    p_detach = sub.add_parser("detach",
                              help="remove ONE replication edge (upstream→downstream); "
                                   "other edges survive")
    g = p_detach.add_argument_group("edge (NAME from config file, or NAME=URI inline)")
    g.add_argument("--downstream", required=True, metavar="NAME[=URI]",
                   help="the standby to detach. Its upstream is auto-discovered from "
                        "its own replicate config (a standby has exactly one), so "
                        "--upstream is optional")
    g.add_argument("--upstream", default=None, metavar="NAME[=URI]",
                   help="the edge's source — optional; pass it to assert WHICH edge "
                        "you mean. (--upstream alone is deliberately NOT accepted: "
                        "detaching ALL of a primary's standbys in one go is never "
                        "implicit.)")

    def _backup_common(p, cluster_required=True, cluster_help="the cluster (config NAME or NAME=URI)"):
        p.add_argument("--cluster", required=cluster_required, metavar="NAME[=URI]",
                       help=cluster_help)
        g = p.add_argument_group("milvus-backup (optional if set in ~/.ternctl.yaml)")
        g.add_argument("--backup-bin", default=None)
        g.add_argument("--backup-workdir", default=None)
        g.add_argument("--backup-config", default=None, metavar="PATH",
                       help="milvus-backup config; default: the --cluster's backup_config")
        return g

    p_backup = sub.add_parser("backup",
                              help="manage backups: create / list / get / restore / delete")
    bsub = p_backup.add_subparsers(dest="backup_command", required=True)
    b_create = bsub.add_parser("create",
                               help="snapshot a cluster (e.g. before reinstalling it)")
    g = _backup_common(b_create, cluster_help="the cluster to back up")
    g.add_argument("--backup-name", "-n", required=True, help="name for the backup")
    g.add_argument("--no-backup-index-extra", dest="backup_index_extra", action="store_false")
    g.add_argument("--backup-create-extra", nargs=argparse.REMAINDER, default=[])
    b_list = bsub.add_parser("list",
                             help="list backups in the archive (--detail adds meta)")
    _backup_common(b_list, cluster_required=False,
                   cluster_help="config-file cluster whose backup_config archive to list")
    b_list.add_argument("--all", action="store_true",
                        help="list the WHOLE archive (disable the --cluster source "
                             "filter); origins are annotated per backup")
    b_list.add_argument("--detail", action="store_true",
                        help="also read each backup's meta: size / milvus version / "
                             "state / collections (one extra archive read per backup)")
    b_get = bsub.add_parser("get", help="show one backup's meta")
    _backup_common(b_get, cluster_required=False)
    b_get.add_argument("--backup-name", "-n", required=True, help="backup to inspect")
    b_restore = bsub.add_parser("restore",
                                help="restore a snapshot into an INDEPENDENT cluster (rollback / clone)")
    g = _backup_common(b_restore, cluster_help="the cluster to restore into")
    g.add_argument("--backup-name", "-n", required=True, help="name of the backup to restore")
    g.add_argument("--restore-suffix", default=None, metavar="SUFFIX",
                   help="restore into NEW collections named <original><suffix> "
                        "(leaves the originals untouched — safest for rollback/compare)")
    g.add_argument("--restore-index", action="store_true", help="also rebuild indexes")
    g.add_argument("--restore-extra", nargs=argparse.REMAINDER, default=[])
    b_delete = bsub.add_parser("delete", help="delete one backup from the archive")
    _backup_common(b_delete, cluster_required=False)
    b_delete.add_argument("--backup-name", "-n", required=True, help="backup to delete")
    b_delete.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    p_clusters = sub.add_parser("clusters",
                                help="list clusters from the config file (--probe checks reachability)")
    p_clusters.add_argument("--probe", action="store_true",
                            help="also probe each cluster's uri for gRPC reachability "
                                 "(transport-level, ~2s timeout per cluster)")

    p_config = sub.add_parser("config", help="manage the cluster config file (~/.ternctl.yaml)")
    csub = p_config.add_subparsers(dest="config_command", required=True)
    c_add = csub.add_parser("add", help="add or update a cluster")
    c_add.add_argument("name")
    c_add.add_argument("--uri", required=True, metavar="URL", help="milvus proxy URI you dial")
    c_add.add_argument("--inter", default=None, metavar="URL",
                       help="inter-cluster URI the OTHER cluster uses to reach this one")
    c_add.add_argument("--token", default=None)
    c_add.add_argument("--pchannel-num", type=int, default=None)
    c_add.add_argument("--cdc-metrics", default=None, metavar="URL",
                       help="this cluster's CDC pod /metrics endpoint (for status lag)")
    c_add.add_argument("--backup-config", default=None, metavar="PATH",
                       help="this cluster's milvus-backup config yaml (lets rebuild/"
                            "backup/restore/backups omit --backup-config*)")
    c_add.add_argument("--kafka-brokers", default=None, metavar="HOSTS",
                       help="this cluster's Kafka bootstrap hosts (lets salvage omit "
                            "--kafka-brokers)")
    c_def = csub.add_parser("set-defaults",
                            help="set environment-wide defaults (milvus-backup bin / workdir)")
    c_def.add_argument("--backup-bin", default=None, metavar="PATH")
    c_def.add_argument("--backup-workdir", default=None, metavar="PATH")
    c_list = csub.add_parser("list", help="list configured clusters")
    c_show = csub.add_parser("show", help="print the raw config file (YAML)")
    c_rm = csub.add_parser("remove", help="remove a cluster")
    c_rm.add_argument("name")

    p_replay = sub.add_parser("replay",
                               help="reconcile a salvage dump into any WRITABLE cluster "
                                    "(fill gaps only; conflicts reported, never guessed)")
    p_replay.add_argument("--from-dir", required=True, metavar="DIR",
                          help="directory of salvage *.jsonl files (ternctl salvage --output-dir)")
    p_replay.add_argument("--into", required=True, metavar="NAME[=URI]",
                          help="ANY writable cluster to recover the stranded rows into — "
                               "typically the new primary; a scratch cluster for "
                               "inspection is equally valid. Standbys are refused "
                               "(read-only).")
    p_replay.add_argument("--collections", default=None, metavar="A,B",
                          help="restrict replay to these collections")
    p_replay.add_argument("--dry-run", action="store_true",
                          help="classify and report only — no writes, no prompt")
    p_replay.add_argument("--overwrite", action="store_true",
                          help="DANGEROUS: the dump wins over the target — existing keys "
                               "are overwritten / deleted per the dump. Only when you KNOW "
                               "no post-failover write touched these keys.")
    p_replay.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    p_salvage = sub.add_parser("salvage",
                               help="recover WAL messages from Kafka using a salvage checkpoint")
    p_salvage.add_argument("--source-cluster", default=None, metavar="NAME",
                           help="config-file name of the OLD PRIMARY. Without "
                                "--source-pchannel this sweeps ALL its pchannels "
                                "(dml_0..N-1) in one run — one output file each — and "
                                "takes kafka brokers from the cluster's kafka_brokers "
                                "config field")
    p_salvage.add_argument("--source-pchannel", default=None, metavar="TOPIC",
                           help="a single SOURCE Kafka topic to read (e.g. "
                                "cluster-a-rootcoord-dml_0); omit to sweep all "
                                "pchannels of --source-cluster")
    p_salvage.add_argument("--kafka-brokers", default=None, metavar="HOSTS",
                           help="source Kafka brokers, e.g. host1:9092,host2:9092 "
                                "(default: --source-cluster's kafka_brokers config field)")
    p_salvage.add_argument("--output-dir", default=None, metavar="DIR",
                           help="sweep mode: directory for per-pchannel jsonl files "
                                "(salvage_dml_<i>.jsonl)")
    p_salvage.add_argument("--checkpoint-file", default=None, metavar="PATH",
                           help="salvage checkpoint JSON from `ternctl force-promote "
                                "--salvage-from`. RECOMMENDED — works after "
                                "force-promote when live GetReplicateInfo is broken "
                                "(milvus-io/milvus#50344).")
    p_salvage.add_argument("--from-offset", type=int, default=None,
                           help="override start offset (default: salvage_checkpoint + 1)")
    p_salvage.add_argument("--new-primary-uri", default=None, metavar="URI",
                           help="live path only: milvus URI holding the salvage checkpoint")
    p_salvage.add_argument("--source-cluster-id", default=None,
                           help="live path only: cluster_id of the OLD primary")
    p_salvage.add_argument("--output", default=None, metavar="PATH",
                           help="JSON Lines output path (required unless --summary-only)")
    p_salvage.add_argument("--summary-only", action="store_true",
                           help="print type breakdown + time range + count only, no jsonl")
    p_salvage.add_argument("--max-msgs", type=int, default=100000,
                           help="cap on messages to dump (default: 100,000)")
    p_salvage.add_argument("--timeout-seconds", type=int, default=10,
                           help="stop after this many idle seconds (default: 10)")
    p_salvage.add_argument("--kafka-sasl-mechanism", default=None,
                           help="PLAIN / SCRAM-SHA-256 / SCRAM-SHA-512")
    p_salvage.add_argument("--kafka-sasl-user", default=None)
    p_salvage.add_argument("--kafka-sasl-password", default=None)
    p_salvage.add_argument("--kafka-ssl", action="store_true",
                           help="enable TLS (SSL, or SASL_SSL when --kafka-sasl-* is set)")

    sub.add_parser("repl", help="enter an interactive shell (run subcommands without "
                                "re-typing 'ternctl' each time)")

    return parser


def run_command(args, parser):
    """Dispatch a single parsed command. Used by both main() and the REPL.
    Lets sys.exit / exceptions propagate so the REPL can catch them."""
    if args.command == "repl":
        run_repl(parser)
        return
    try:
        if args.command == "force-promote":
            config = load_config(getattr(args, "config", None))
            target = resolve_cluster("target", args.target, config,
                                     inter=args.target_inter, token=getattr(args, "token", None),
                                     pchannel_num=getattr(args, "pchannel_num", None))
            do_force_promote(args, target)
            return

        if args.command == "clusters":
            do_clusters(args)
            return

        if args.command == "config":
            do_config(args)
            return

        if args.command == "salvage":
            do_salvage(args)
            return

        if args.command == "replay":
            config = load_config(getattr(args, "config", None))
            into = resolve_cluster("into", args.into, config, token=getattr(args, "token", None))
            do_replay(args, into)
            return

        if args.command == "backup":
            # --backup-config / bin / workdir resolvable from the config file.
            if not args.cluster and not args.backup_config:
                raise RuntimeError("give --cluster NAME (config file) or --backup-config PATH")
            config = load_config(getattr(args, "config", None))
            cid = (args.cluster or "").split("=", 1)[0].strip() or None
            fill_backup_args(args, config, up_cid=cid)
            {"create": do_backup, "list": do_backups, "get": do_backup_get,
             "restore": do_restore, "delete": do_backup_delete}[args.backup_command](args)
            return

        if args.command == "topology":
            do_topology(args)
            return

        if args.command == "switchover":
            do_switchover(args, load_config(getattr(args, "config", None)))
            return

        if args.command == "detach":
            config = load_config(getattr(args, "config", None))
            downstream = resolve_cluster("downstream", args.downstream, config,
                                         token=getattr(args, "token", None), pchannel_num=getattr(args, "pchannel_num", None))
            if args.upstream:
                upstream = resolve_cluster("upstream", args.upstream, config,
                                           token=getattr(args, "token", None), pchannel_num=getattr(args, "pchannel_num", None))
            else:
                upstream = discover_upstream(args, downstream, config)
            do_detach(args, upstream, downstream)
            return

        if args.command in ("status", "verify") and not (args.upstream and args.downstream):
            config = load_config(getattr(args, "config", None))
            fn = do_status if args.command == "status" else verify
            if args.upstream:
                upstream = resolve_cluster("upstream", args.upstream, config,
                                           inter=args.upstream_inter, token=getattr(args, "token", None),
                                           pchannel_num=getattr(args, "pchannel_num", None))
                if args.command == "verify":
                    ids, downstreams = discover_downstreams(args, upstream, config)
                    if not ids:
                        warn(f"{upstream.cluster_id} has no outgoing replication "
                             f"edges — nothing to verify")
                        ok = True
                    else:
                        info(f"downstreams of {upstream.cluster_id} (from its "
                             f"replicate config): " + ", ".join(ids))
                        ok = bool(downstreams) and verify_many(args, upstream, downstreams)
                else:
                    ok = for_each_downstream(args, upstream, config, fn)
            elif args.downstream:
                downstream = resolve_cluster("downstream", args.downstream, config,
                                             inter=args.downstream_inter, token=getattr(args, "token", None),
                                             pchannel_num=getattr(args, "pchannel_num", None))
                upstream = discover_upstream(args, downstream, config)
                r = fn(args, upstream, downstream)
                ok = r is None or bool(r)
            else:
                raise RuntimeError(
                    "give --upstream and/or --downstream — either side can be "
                    "inferred from the other")
            if args.command == "verify":
                sys.exit(0 if ok else 1)
            return

        upstream, downstream = clusters_from_args(args)
        if args.command == "rebuild":
            config = load_config(getattr(args, "config", None))
            fill_backup_args(args, config,
                             up_cid=upstream.cluster_id, down_cid=downstream.cluster_id)
            do_rebuild(args, upstream, downstream)
        elif args.command == "status":
            do_status(args, upstream, downstream)
        elif args.command == "verify":
            ok = verify(args, upstream, downstream)
            sys.exit(0 if ok else 1)
        elif args.command == "attach":
            do_attach(args, upstream, downstream)

    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)


def _subparser_choices(parser):
    """Map of subcommand name -> its subparser, for `help <cmd>` in the REPL."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


# Flags whose VALUE is a cluster name from the config file.
_CLUSTER_VALUE_FLAGS = {"--cluster", "--clusters", "--upstream", "--downstream",
                        "--target", "--source-cluster", "--into"}
# Flags whose VALUE is a filesystem path → complete from the filesystem.
_PATH_VALUE_FLAGS = {"--checkpoint-file", "--output", "--output-dir", "--from-dir",
                     "--backup-bin", "--backup-workdir",
                     "--backup-config", "--backup-config-secondary"}


def _path_candidates(cur):
    """Filesystem completion: directories get a trailing '/' (keep typing),
    files get the usual trailing space."""
    import glob
    import os
    pattern = os.path.expanduser(cur) + "*"
    out = []
    for p in sorted(glob.glob(pattern)):
        out.append(p + "/" if os.path.isdir(p) else p + " ")
    return out


def _completion_candidates(parser, buf):
    """Pure completion logic for the REPL: given the line buffer up to the
    cursor, return the candidate completions for the word being typed.
    Kept readline-free so it can be unit-tested directly.

    Layers:
      1. first word            -> subcommand names (+ help / exit)
      2. after `help`          -> subcommand names
      3. after a subcommand    -> that subparser's --flags
      4. after a cluster flag  -> cluster names from ~/.ternctl.yaml
         (--clusters also completes each element of a comma-list)
    """
    choices = _subparser_choices(parser)
    try:
        toks = buf.split()
    except Exception:
        return []
    at_new_word = buf.endswith(" ") or not buf
    cur = "" if at_new_word else (toks[-1] if toks else "")
    prev_toks = toks if at_new_word else toks[:-1]

    def cluster_names():
        try:
            return sorted(load_config(None).keys())
        except Exception:
            return []

    if not prev_toks:
        cands = sorted(choices) + ["help", "exit"]
    elif prev_toks[0] in ("help", "?", "h"):
        cands = sorted(choices)
    elif prev_toks[0] in choices:
        prev = prev_toks[-1]
        sub = choices[prev_toks[0]]
        nested = _subparser_choices(sub)  # e.g. backup create/list/…, config add/…
        if prev in _PATH_VALUE_FLAGS or (not cur.startswith("-") and
                                          cur[:1] in ("/", ".", "~")):
            return _path_candidates(cur)
        if prev in _CLUSTER_VALUE_FLAGS and not cur.startswith("-"):
            names = cluster_names()
            head, sep, tail = cur.rpartition(",")
            if sep:  # completing an element of a comma-list: a,b,<TAB>
                # No trailing space: the user may keep extending the list with ','.
                used = set(head.split(","))
                return [head + "," + n for n in names
                        if n.startswith(tail) and n != tail and n not in used]
            cands = names
        elif nested and len(prev_toks) == 1:
            cands = sorted(nested)  # second word = the subcommand
        else:
            # flags come from the nested subparser when one is in play
            if nested and len(prev_toks) >= 2 and prev_toks[1] in nested:
                sub = nested[prev_toks[1]]
            cands = sorted({o for a in sub._actions for o in a.option_strings
                            if o.startswith("--")})
    else:
        cands = []
    # Trailing space: python readline can't set rl_completion_append_character,
    # so bake the separator into the candidate (a fully-completed word is always
    # followed by more input — a flag, a value, or another argument).
    return [c + " " for c in cands if c.startswith(cur) and c != cur]


def _make_completer(parser):
    """readline completer closure over _completion_candidates."""
    def complete(text, state):
        try:
            import readline
            buf = readline.get_line_buffer()[:readline.get_endidx()]
            matches = _completion_candidates(parser, buf)
            return matches[state] if state < len(matches) else None
        except Exception:
            return None  # never let completion errors kill the REPL
    return complete


def run_repl(parser):
    """Interactive shell: run subcommands without re-typing the launcher each time.
    Uses stdlib only — readline (if present) gives history + line editing +
    tab completion (subcommands, flags, cluster names from the config file)."""
    import shlex
    try:
        import readline
        # Words are space-separated only — keeps '--flag' and 'a,b,c' whole.
        readline.set_completer_delims(" \t\n")
        readline.set_completer(_make_completer(parser))
        if "libedit" in (getattr(readline, "__doc__", "") or ""):
            readline.parse_and_bind("bind ^I rl_complete")   # macOS libedit
        else:
            readline.parse_and_bind("tab: complete")
    except ImportError:
        pass
    print(f"\n{_bold(_cyan('ternctl interactive shell'))}")
    print(_dim("  run subcommands directly (status, topology, rebuild, config list, ...)"))
    print(_dim("  TAB completes commands, flags and cluster names · 'help' lists commands · 'exit' or Ctrl-D quits\n"))
    while True:
        output._spinner_end()  # belt-and-braces: never let a dangling spinner draw over the prompt
        try:
            line = input("ternctl> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("  (use 'exit' or Ctrl-D to quit)")
            continue
        if not line:
            continue
        if line in ("exit", "quit", "q"):
            break
        try:
            argv = shlex.split(line)
        except ValueError as e:
            print(f"  {_red('parse error')}: {e}")
            continue
        if argv and argv[0] in ("help", "?", "h"):
            # `help` → top-level usage; `help <subcommand>` → that command's help.
            choices = _subparser_choices(parser)
            if len(argv) > 1 and argv[1] in choices:
                choices[argv[1]].print_help()
            elif len(argv) > 1:
                print(f"  {_red('no such command')}: {argv[1]} "
                      f"(try {_bold('help')} for the list)")
            else:
                parser.print_help()
            continue
        if argv and argv[0] == "repl":
            print("  already in the shell")
            continue
        try:
            sub_args = parser.parse_args(argv)
        except SystemExit:
            continue  # argparse printed an error or --help; stay in the shell
        try:
            run_command(sub_args, parser)
        except SystemExit:
            pass  # a subcommand called sys.exit(); don't kill the shell
        except Exception as e:
            print(f"  {_red('error')}: {e}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    output._VERBOSE = getattr(args, "verbose", False)
    if getattr(args, "no_color", False):
        output._NO_COLOR = True
    run_command(args, parser)


if __name__ == "__main__":
    main()
