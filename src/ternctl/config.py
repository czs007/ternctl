"""Cluster config file (~/.ternctl.yaml) + spec resolution (name | name=uri)."""
import os

from .cluster import Cluster


# Cluster config file (~/.ternctl.yaml) — kubectl-style: define clusters once,
# reference them by name. Inline `name=uri` specs still work without a config.
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.ternctl.yaml")
# Per-cluster fields stored in the config file.
CONFIG_FIELDS = ("uri", "inter_uri", "token", "pchannel_num", "cdc_metrics")


def config_path(override=None):
    return override or os.environ.get("TERNCTL_CONFIG") or DEFAULT_CONFIG_PATH


def load_config(path=None):
    """Return {cluster_name: {uri, inter_uri, token, pchannel_num, cdc_metrics}}.
    Empty dict if the file doesn't exist. PyYAML is imported lazily so inline
    `name=uri` specs work even without it installed."""
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
        doc = yaml.safe_load(f) or {}
    return doc.get("clusters", {}) or {}


def save_config(clusters, path=None):
    import yaml
    p = config_path(path)
    with open(p, "w") as f:
        yaml.safe_dump({"clusters": clusters}, f, sort_keys=True, default_flow_style=False)
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


