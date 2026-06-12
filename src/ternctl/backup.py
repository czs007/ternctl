"""milvus-backup CLI integration (backup create / restore secondary / list)."""
import json
import os
import re
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
    os.makedirs(args.backup_workdir, exist_ok=True)
    if output._VERBOSE:
        result = subprocess.run(cmd, cwd=args.backup_workdir)
    else:
        log_path = os.path.join(args.backup_workdir, "milvus-backup-cli.log")
        with open(log_path, "ab") as f:
            f.write(("\n==== " + " ".join(argv) + " ====\n").encode())
            f.flush()  # subprocess writes the SAME fd directly — without this
                       # flush the buffered header can land AFTER its output,
                       # scrambling per-run sections in the log
            result = subprocess.run(cmd, cwd=args.backup_workdir, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        log_path = os.path.join(args.backup_workdir, "milvus-backup-cli.log")
        tail = "" if output._VERBOSE else _error_tail(log_path)
        if not output._VERBOSE:
            warn(f"full milvus-backup output: {log_path}")
        raise RuntimeError(
            f"milvus-backup exited with code {result.returncode}: {' '.join(argv)}"
            + (f"\n  cause: {tail}" if tail else ""))


def _error_tail(log_path):
    """The actual failure line(s) from the CURRENT run's section of the log —
    the operator should never have to dig the log file for the root cause."""
    try:
        text = open(log_path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return ""
    section = text.rsplit("\n==== ", 1)[-1]          # this invocation only
    lines = [l.strip() for l in section.splitlines() if l.strip()]
    hits = [l for l in lines
            if re.search(r"(?i)\berror\b|invalid|failed|cannot|denied|refused|exceed", l)]
    return " | ".join((hits or lines)[-2:])


def run_backup_capture(args, argv):
    """Run milvus-backup and CAPTURE its stdout — for commands whose output IS
    the result (list / get), unlike run_backup which logs it away."""
    cmd = [os.path.abspath(args.backup_bin)] + argv
    os.makedirs(args.backup_workdir, exist_ok=True)
    result = subprocess.run(cmd, cwd=args.backup_workdir,
                            capture_output=True, text=True)
    if result.returncode != 0:
        tail = ((result.stdout or "") + (result.stderr or ""))[-400:]
        raise RuntimeError(
            f"milvus-backup exited with code {result.returncode}: "
            f"{' '.join(argv)}\n{tail}")
    return result.stdout or ""


def backup_list_names(args):
    """Names of all backups in the archive of this --backup-config.
    Parses the `>> Backups:` block that milvus-backup `list` prints after
    its log lines."""
    out = run_backup_capture(args, ["--config", args.backup_config, "list"])
    names, in_block = [], False
    for line in out.splitlines():
        if in_block:
            line = line.strip()
            if line:
                names.append(line)
        elif line.strip().startswith(">> Backups"):
            in_block = True
    return names


def backup_get_info(args, name):
    """Parsed JSON from milvus-backup `get -n <name>` (None if unparsable)."""
    out = run_backup_capture(args, ["--config", args.backup_config, "get", "-n", name])
    i = out.find("{")
    if i < 0:
        return None
    try:
        return json.loads(out[i:])
    except ValueError:
        return None


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
