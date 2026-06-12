"""Cluster config file (~/.ternctl.yaml) + spec resolution (name | name=uri)."""
import os

from .cluster import Cluster


# Cluster config file (~/.ternctl.yaml) — kubectl-style: define clusters once,
# reference them by name. Inline `name=uri` specs still work without a config.
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.ternctl.yaml")
# Per-cluster fields stored in the config file.
CONFIG_FIELDS = ("uri", "inter_uri", "token", "pchannel_num", "cdc_metrics",
                 "backup_config", "kafka_brokers")
# Environment-wide fields stored under the top-level `defaults:` key.
DEFAULT_FIELDS = ("backup_bin", "backup_workdir", "backup_config")


def config_path(override=None):
    return override or os.environ.get("TERNCTL_CONFIG") or DEFAULT_CONFIG_PATH


def _load_doc(path=None):
    p = config_path(path)
    if not os.path.exists(p):
        return {}
    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            f"reading {p} needs PyYAML (`pip install pyyaml`) — or skip the config "
            f"file and pass clusters inline as name=uri")
    with open(p) as f:
        return yaml.safe_load(f) or {}


def load_config(path=None):
    """Return {cluster_name: {uri, inter_uri, token, pchannel_num, cdc_metrics,
    backup_config, kafka_brokers}}. Empty dict if the file doesn't exist.
    PyYAML is imported lazily so inline `name=uri` specs work without it."""
    return _load_doc(path).get("clusters", {}) or {}


def load_defaults(path=None):
    """Environment-wide defaults ({backup_bin, backup_workdir}) from the
    top-level `defaults:` key. Empty dict if absent."""
    return _load_doc(path).get("defaults", {}) or {}


def save_config(clusters, path=None, defaults=None):
    """Write the clusters map (and optionally replace defaults), preserving any
    other top-level keys already in the file."""
    import yaml
    p = config_path(path)
    doc = _load_doc(path) if os.path.exists(p) else {}
    doc["clusters"] = clusters
    if defaults is not None:
        doc["defaults"] = defaults
    with open(p, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=True, default_flow_style=False)
    return p


def resolve_cluster(role, spec, config, inter=None, token=None, pchannel_num=None,
                    grpc=None, cdc_metrics=None):
    """Resolve a cluster spec into a Cluster.

    spec is either:
      - "name"      → looked up in the config file
      - "name=uri"  → inline (no config needed)

    Single-flag overrides (inter/token/...) win over the config-file values.
    """
    if "=" in spec:
        cid, uri = spec.split("=", 1)
        cid, uri = cid.strip(), uri.strip()
        entry = {}
    else:
        cid = spec.strip()
        if cid not in config:
            known = ", ".join(sorted(config)) or "(config file empty or missing)"
            raise RuntimeError(
                f"cluster '{cid}' is not in the config file and not an inline "
                f"name=uri spec.\n  known clusters: {known}\n  add it:  ternctl "
                f"config add {cid} --uri http://...:19530 [--inter http://...]\n"
                f"  or inline:  --{role} {cid}=http://...:19530")
        entry = config[cid]
        uri = entry.get("uri")
        if not uri:
            raise RuntimeError(f"cluster '{cid}' in the config file has no 'uri'")
    return Cluster(
        role, uri, cid,
        pchannel_num if pchannel_num is not None else int(entry.get("pchannel_num", 16)),
        token or entry.get("token") or "root:Milvus",
        inter_uri=inter or entry.get("inter_uri"),
        grpc_override=grpc or entry.get("grpc"),
        cdc_metrics=cdc_metrics or entry.get("cdc_metrics"),
    )


