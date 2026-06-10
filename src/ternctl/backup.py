"""milvus-backup CLI integration (backup create / restore secondary)."""
import os
import subprocess

from . import output
from .output import warn


# --------------------------------------------------------------------------- #
def run_backup(args, argv):
    """Run milvus-backup CLI. In demo mode (default), its stdout/stderr are
    redirected to a log file inside backup_workdir so the demo screen stays
    clean. Pass --verbose to stream them through.
    """
    cmd = [os.path.abspath(args.backup_bin)] + argv
    if output._VERBOSE:
        result = subprocess.run(cmd, cwd=args.backup_workdir)
    else:
        os.makedirs(args.backup_workdir, exist_ok=True)
        log_path = os.path.join(args.backup_workdir, "milvus-backup-cli.log")
        with open(log_path, "ab") as f:
            f.write(("\n==== " + " ".join(argv) + " ====\n").encode())
            result = subprocess.run(cmd, cwd=args.backup_workdir, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        if not output._VERBOSE:
            warn(f"see {os.path.join(args.backup_workdir, 'milvus-backup-cli.log')} for milvus-backup output")
        raise RuntimeError(f"milvus-backup exited with code {result.returncode}: {' '.join(argv)}")


def backup_create(args):
    argv = ["--config", args.backup_config, "create", "-n", args.backup_name]
    if args.backup_index_extra:
        argv.append("--backup_index_extra")
    argv += args.backup_create_extra
    run_backup(args, argv)


def restore_secondary(args, upstream, downstream):
    argv = [
        "--config", args.backup_config_secondary,
        "restore", "secondary",
        "-n", args.backup_name,
        "--source_cluster_id", upstream.cluster_id,
        "--target_cluster_id", downstream.cluster_id,
    ]
    run_backup(args, argv)


def restore_backup(args):
    """Plain (non-secondary) restore of a backup into a cluster — rollback / clone.
    With --restore-suffix the originals are left untouched (restored into new
    collections <name><suffix>)."""
    argv = ["--config", args.backup_config, "restore", "-n", args.backup_name]
    if getattr(args, "restore_suffix", None):
        argv += ["-s", args.restore_suffix]
    if getattr(args, "restore_index", False):
        argv.append("--restore_index")
    argv += getattr(args, "restore_extra", [])
    run_backup(args, argv)


# --------------------------------------------------------------------------- #
# pymilvus verification
# --------------------------------------------------------------------------- #
