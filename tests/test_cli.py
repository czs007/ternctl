"""Smoke tests for the argument parser (no dispatch / no live cluster)."""
import pytest

from ternctl.cli import build_parser


def test_parser_builds():
    p = build_parser()
    assert p.prog == "ternctl"


def test_status_inline_parses():
    p = build_parser()
    args = p.parse_args(["status",
                         "--upstream", "cluster-a=http://127.0.0.1:19530",
                         "--downstream", "cluster-b=http://127.0.0.1:19531"])
    assert args.command == "status"
    assert args.upstream == "cluster-a=http://127.0.0.1:19530"


def test_up_cdc_alias():
    """--up-cdc is an alias for --upstream-cdc-metrics."""
    p = build_parser()
    a1 = p.parse_args(["status", "--upstream", "a=u", "--downstream", "b=u",
                       "--up-cdc", "http://127.0.0.1:9091"])
    a2 = p.parse_args(["status", "--upstream", "a=u", "--downstream", "b=u",
                       "--upstream-cdc-metrics", "http://127.0.0.1:9091"])
    assert a1.upstream_cdc_metrics == a2.upstream_cdc_metrics == "http://127.0.0.1:9091"


def test_force_promote_target():
    p = build_parser()
    args = p.parse_args(["force-promote", "--target", "cluster-b", "--yes"])
    assert args.command == "force-promote"
    assert args.target == "cluster-b"
    assert args.yes is True


def test_config_add_parses():
    p = build_parser()
    args = p.parse_args(["config", "add", "cluster-a", "--uri", "http://127.0.0.1:19530"])
    assert args.config_command == "add"
    assert args.name == "cluster-a"


def test_salvage_parses():
    p = build_parser()
    args = p.parse_args(["salvage",
                         "--source-pchannel", "cluster-a-rootcoord-dml_0",
                         "--kafka-brokers", "localhost:9092",
                         "--from-checkpoint-file", "cp.json",
                         "--output", "out.jsonl"])
    assert args.command == "salvage"
    assert args.source_pchannel == "cluster-a-rootcoord-dml_0"
    assert args.from_checkpoint_file == "cp.json"


def test_backup_parses():
    p = build_parser()
    args = p.parse_args(["backup", "--cluster", "cluster-a",
                         "--backup-name", "bk", "--backup-config", "backup-a.yaml"])
    assert args.command == "backup"
    assert args.cluster == "cluster-a"
    assert args.backup_name == "bk"


def test_restore_parses():
    p = build_parser()
    args = p.parse_args(["restore", "--cluster", "cluster-c",
                         "--backup-name", "bk", "--backup-config", "backup-c.yaml",
                         "--restore-suffix", "_r"])
    assert args.command == "restore"
    assert args.cluster == "cluster-c"
    assert args.restore_suffix == "_r"


def test_subcommand_required():
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])
