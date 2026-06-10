"""Argument parser, command dispatch, REPL, and the `ternctl` entry point."""
import argparse
import sys

from . import output
from .output import log, _red, _cyan, _bold, _dim
from .config import load_config, resolve_cluster
from .verify import verify
from .commands import (do_rebuild, do_switchover, do_force_promote, do_status,
                       do_config, do_topology, do_replicate_config, do_break_topology,
                       do_backup)
from .salvage import do_salvage


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_common(p):
    g = p.add_argument_group("clusters (NAME from config file, or NAME=URI inline)")
    g.add_argument("--upstream", required=True, metavar="NAME[=URI]",
                   help="source cluster: a name from ~/.ternctl.yaml, or inline "
                        "NAME=URI (e.g. cluster-a=http://127.0.0.1:19530)")
    g.add_argument("--downstream", required=True, metavar="NAME[=URI]",
                   help="target cluster (same NAME or NAME=URI form)")
    g.add_argument("--upstream-inter", default=None, metavar="URI",
                   help="override the upstream's inter-cluster URI")
    g.add_argument("--downstream-inter", default=None, metavar="URI",
                   help="override the downstream's inter-cluster URI")
    g.add_argument("--pchannel-num", type=int, default=None,
                   help="override pchannel count (default: config value or 16)")
    g.add_argument("--token", default=None, help="override auth token")
    g.add_argument("--config", default=None, metavar="PATH",
                   help="config file path (default ~/.ternctl.yaml)")


def add_backup(p):
    g = p.add_argument_group("milvus-backup")
    g.add_argument("--backup-bin", default="./milvus-backup")
    g.add_argument("--backup-workdir", default=".")
    g.add_argument("--backup-name", default="ternctl_backup")
    g.add_argument("--backup-config", default="backup.yaml")
    g.add_argument("--backup-config-secondary", default="backup.yaml")
    g.add_argument("--no-backup-index-extra", dest="backup_index_extra", action="store_false")
    g.add_argument("--backup-create-extra", nargs=argparse.REMAINDER, default=[])


def clusters_from_args(args):
    config = load_config(getattr(args, "config", None))
    upstream = resolve_cluster("upstream", args.upstream, config,
                               inter=args.upstream_inter, token=args.token,
                               pchannel_num=args.pchannel_num)
    downstream = resolve_cluster("downstream", args.downstream, config,
                                 inter=args.downstream_inter, token=args.token,
                                 pchannel_num=args.pchannel_num)
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

    p_switch = sub.add_parser("switchover", help="reverse topology (graceful promote)")
    add_common(p_switch)

    p_force = sub.add_parser("force-promote", help="promote secondary to standalone primary (original primary down)")
    g = p_force.add_argument_group("target (NAME from config, or NAME=URI inline)")
    g.add_argument("--target", required=True, metavar="NAME[=URI]",
                   help="the secondary being promoted: a config name, or NAME=URI inline")
    g.add_argument("--target-inter", default=None, metavar="URI",
                   help="override the target's inter-cluster URI")
    g.add_argument("--pchannel-num", type=int, default=None)
    g.add_argument("--token", default=None)
    g.add_argument("--config", default=None, metavar="PATH",
                   help="config file path (default ~/.ternctl.yaml)")
    p_force.add_argument("--yes", action="store_true", help="skip the RPO confirmation prompt")
    sg = p_force.add_argument_group("salvage checkpoint prefetch (recommended for DR)")
    sg.add_argument("--salvage-source-cluster-id", default=None,
                    help="cluster id of the OLD primary you may want to salvage data from. "
                         "When set, the tool snapshots the live ReplicateCheckpoint for every "
                         "target pchannel BEFORE the force_promote RPC, while GetReplicateInfo "
                         "still works. After force_promote that API is broken on a standalone "
                         "primary — see milvus-io/milvus#50344. Without this flag, salvage of "
                         "in-flight messages from the dead primary is not possible.")
    sg.add_argument("--salvage-output", default=None,
                    help="output path for the prefetched checkpoint JSON. "
                         "Default: ./salvage_checkpoint_<target>_<unix_ts>.json")

    p_status = sub.add_parser("status", help="dump replication checkpoints")
    add_common(p_status)
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
    p_topo.add_argument("--clusters", default=None, metavar="N1,N2,N3",
                        help="comma-separated cluster names from the config file "
                             "(shorthand for repeating --cluster)")
    p_topo.add_argument("--pchannel-num", type=int, default=None)
    p_topo.add_argument("--token", default=None)
    p_topo.add_argument("--config", default=None, metavar="PATH",
                        help="config file path (default ~/.ternctl.yaml)")

    p_verify = sub.add_parser("verify", help="compare row counts")
    add_common(p_verify)
    p_verify.add_argument("--collections", default=None)
    p_verify.add_argument("--once", action="store_true",
                          help="single snapshot — skip the internal retry/wait loop. "
                               "Shows both sides' current counts and, if they differ, "
                               "how far behind the target is, without declaring FAILED. "
                               "Re-run it (or `watch`) to watch the standby converge "
                               "while the source is being written.")

    p_repl = sub.add_parser("replicate-config", help="raw inittarget equivalent")
    add_common(p_repl)
    p_repl.add_argument("--direction", choices=["up2down", "down2up"], default="up2down")
    p_repl.add_argument("--target", choices=["upstream", "downstream", "both"], default="both")

    p_break = sub.add_parser("break-topology",
                             help="delete the replication edge between two clusters (cleanup/teardown)")
    add_common(p_break)

    p_backup = sub.add_parser("backup",
                              help="snapshot a single cluster via milvus-backup (e.g. before reinstalling it)")
    p_backup.add_argument("--cluster", required=True, metavar="NAME[=URI]",
                          help="the cluster to back up (config NAME or NAME=URI)")
    p_backup.add_argument("--config", default=None, metavar="PATH",
                          help="ternctl config file (default ~/.ternctl.yaml)")
    bg = p_backup.add_argument_group("milvus-backup")
    bg.add_argument("--backup-bin", default="./milvus-backup")
    bg.add_argument("--backup-workdir", default=".")
    bg.add_argument("--backup-name", required=True, help="name for the backup")
    bg.add_argument("--backup-config", required=True, metavar="PATH",
                    help="milvus-backup config pointing at this cluster (its milvus / minio / etcd)")
    bg.add_argument("--no-backup-index-extra", dest="backup_index_extra", action="store_false")
    bg.add_argument("--backup-create-extra", nargs=argparse.REMAINDER, default=[])

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
    c_add.add_argument("--config", default=None, metavar="PATH")
    c_list = csub.add_parser("list", help="list configured clusters")
    c_list.add_argument("--config", default=None, metavar="PATH")
    c_show = csub.add_parser("show", help="print the raw config file (YAML)")
    c_show.add_argument("--config", default=None, metavar="PATH")
    c_rm = csub.add_parser("remove", help="remove a cluster")
    c_rm.add_argument("name")
    c_rm.add_argument("--config", default=None, metavar="PATH")

    p_salvage = sub.add_parser("salvage",
                               help="recover WAL messages from Kafka using a salvage checkpoint")
    p_salvage.add_argument("--source-pchannel", required=True, metavar="TOPIC",
                           help="the SOURCE Kafka topic to read (= the old primary's pchannel "
                                "name, e.g. cluster-a-rootcoord-dml_0)")
    p_salvage.add_argument("--kafka-brokers", required=True, metavar="HOSTS",
                           help="source Kafka brokers, e.g. host1:9092,host2:9092")
    p_salvage.add_argument("--from-checkpoint-file", default=None, metavar="PATH",
                           help="salvage checkpoint JSON from `ternctl force-promote "
                                "--salvage-source-cluster-id`. RECOMMENDED — works after "
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
    p_salvage.add_argument("--token", default="root:Milvus",
                           help="milvus auth token for the live checkpoint lookup")
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
                                     inter=args.target_inter, token=args.token,
                                     pchannel_num=args.pchannel_num)
            if not args.yes:
                ans = input(
                    f"\nFORCE-PROMOTE will make '{target.cluster_id}' a standalone primary.\n"
                    f"Data written to the OLD primary after the CDC lag horizon will be LOST.\n"
                    f"Type 'force-promote' to confirm: "
                ).strip()
                if ans != "force-promote":
                    log("aborted")
                    sys.exit(1)
            do_force_promote(args, target)
            return

        if args.command == "config":
            do_config(args)
            return

        if args.command == "salvage":
            do_salvage(args)
            return

        if args.command == "backup":
            do_backup(args)
            return

        if args.command == "topology":
            do_topology(args)
            return

        upstream, downstream = clusters_from_args(args)
        if args.command == "rebuild":
            do_rebuild(args, upstream, downstream)
        elif args.command == "switchover":
            do_switchover(args, upstream, downstream)
        elif args.command == "status":
            do_status(args, upstream, downstream)
        elif args.command == "verify":
            ok = verify(args, upstream, downstream)
            sys.exit(0 if ok else 1)
        elif args.command == "replicate-config":
            do_replicate_config(args, upstream, downstream)
        elif args.command == "break-topology":
            do_break_topology(args, upstream, downstream)
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)


def run_repl(parser):
    """Interactive shell: run subcommands without re-typing the launcher each time.
    Uses stdlib only — readline (if present) gives history + line editing."""
    import shlex
    try:
        import readline  # noqa: F401 — enables up-arrow history + line editing
    except ImportError:
        pass
    print(f"\n{_bold(_cyan('ternctl interactive shell'))}")
    print(_dim("  run subcommands directly (status, topology, rebuild, config list, ...)"))
    print(_dim("  'help' lists them · 'exit' or Ctrl-D quits\n"))
    while True:
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
        if line in ("help", "?"):
            parser.print_help()
            continue
        try:
            argv = shlex.split(line)
        except ValueError as e:
            print(f"  {_red('parse error')}: {e}")
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
