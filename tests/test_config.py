"""Unit tests for cluster-spec resolution (no live cluster needed)."""
import pytest

from ternctl.config import resolve_cluster, load_config


def test_inline_spec():
    """`name=uri` resolves without any config file."""
    c = resolve_cluster("upstream", "cluster-a=http://127.0.0.1:19530", {})
    assert c.cluster_id == "cluster-a"
    assert c.uri == "http://127.0.0.1:19530"
    assert c.role == "upstream"
    assert c.pchannel_num == 16          # default
    assert c.token == "root:Milvus"      # default


def test_reference_spec():
    """A bare name is looked up in the config dict."""
    cfg = {"cluster-b": {"uri": "http://127.0.0.1:19531",
                         "inter_uri": "http://b-internal:19530",
                         "cdc_metrics": "http://127.0.0.1:9091"}}
    c = resolve_cluster("downstream", "cluster-b", cfg)
    assert c.cluster_id == "cluster-b"
    assert c.uri == "http://127.0.0.1:19531"
    assert c.inter_uri == "http://b-internal:19530"
    assert c.cdc_metrics == "http://127.0.0.1:9091"


def test_unknown_name_errors():
    """A name not in config (and not inline) is a clear error, not a crash."""
    with pytest.raises(RuntimeError) as e:
        resolve_cluster("upstream", "nope", {"cluster-a": {"uri": "http://x"}})
    msg = str(e.value)
    assert "nope" in msg
    assert "cluster-a" in msg            # lists known clusters


def test_flag_overrides_config():
    """Single-flag overrides win over config-file values."""
    cfg = {"cluster-c": {"uri": "http://127.0.0.1:19532",
                        "inter_uri": "http://config-inter:19530",
                        "token": "config:token"}}
    c = resolve_cluster("upstream", "cluster-c", cfg,
                        inter="http://override:19530", token="override:token")
    assert c.inter_uri == "http://override:19530"
    assert c.token == "override:token"


def test_config_missing_uri_errors():
    with pytest.raises(RuntimeError):
        resolve_cluster("upstream", "broken", {"broken": {"inter_uri": "http://x"}})


def test_load_config_missing_file_is_empty(tmp_path):
    assert load_config(str(tmp_path / "does-not-exist.yaml")) == {}
