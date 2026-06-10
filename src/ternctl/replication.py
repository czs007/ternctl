"""Replication config build/apply, checkpoint & CDC-latency observers, salvage."""

import grpc
from pymilvus.grpc_gen import common_pb2, milvus_pb2

from .output import (log)
from .cluster import auth_metadata, status_ok


def build_replicate_config(upstream, downstream, source, target):
    topology = common_pb2.CrossClusterTopology(
        source_cluster_id=source.cluster_id,
        target_cluster_id=target.cluster_id,
    )
    return common_pb2.ReplicateConfiguration(
        clusters=[upstream.milvus_cluster(), downstream.milvus_cluster()],
        cross_cluster_topology=[topology],
    )


def apply_replicate_config(target_cluster, config, force_promote=False, _quiet=False):
    """Push an UpdateReplicateConfiguration to one cluster.

    `_quiet=True` suppresses the success line — used when the caller is already
    showing a step()/done() wrapper, so the demo output stays uncluttered.
    """
    stub, channel = target_cluster.stub()
    try:
        req = milvus_pb2.UpdateReplicateConfigurationRequest(
            replicate_configuration=config,
        )
        if force_promote:
            if not hasattr(req, "force_promote"):
                raise RuntimeError(
                    "pymilvus is too old: UpdateReplicateConfigurationRequest "
                    "has no force_promote field. Upgrade pymilvus to a build "
                    "compatible with Milvus 2.6.16+."
                )
            req.force_promote = True
        resp = stub.UpdateReplicateConfiguration(
            req, metadata=auth_metadata(target_cluster.token), timeout=60,
        )
        if not status_ok(resp):
            raise RuntimeError(
                f"UpdateReplicateConfiguration on {target_cluster.role} "
                f"({target_cluster.dial_addr}) failed: {resp.reason or resp}"
            )
        if not _quiet:
            tag = " (force_promote=True)" if force_promote else ""
            log(f"replicate configuration applied on {target_cluster.role} ({target_cluster.dial_addr}){tag}")
    finally:
        channel.close()


def standalone_replicate_config(target_cluster):
    """Build the no-topology, current-cluster-only config that force_promote requires."""
    return common_pb2.ReplicateConfiguration(
        clusters=[target_cluster.milvus_cluster()],
        cross_cluster_topology=[],
    )


def break_topology_config(*clusters):
    """Build a ReplicateConfiguration that keeps the cluster definitions but has
    EMPTY cross_cluster_topology. Applying this is the supported way to delete a
    replication edge ("rewrite the topology to remove the edge").

    Used for cleanup / teardown only. Do NOT use this as a "pause" — the
    source-side WAL retention keeps ticking, so re-creating the edge later may
    silently lose data if retention has expired in between.
    """
    return common_pb2.ReplicateConfiguration(
        clusters=[c.milvus_cluster() for c in clusters],
        cross_cluster_topology=[],
    )


def get_replicate_checkpoints(observer, source_cluster_id, pchannels):
    """Return {pchannel: time_tick}, distinguishing three states:
      - int > 0  : active (CDC has forwarded; this is the last-replicated tick)
      - 0        : idle — RPC succeeded but the checkpoint hasn't advanced yet
                   (replication is configured but no traffic has flowed, e.g.
                   right after a switchover with no writes since)
      - None     : unreachable — the GetReplicateInfo RPC itself failed
    The 0-vs-None split is the whole point: a configured-but-idle edge looks
    identical to a broken one if you collapse both to "n/a".
    """
    stub, channel = observer.stub()
    out = {}
    try:
        for pch in pchannels:
            try:
                resp = stub.GetReplicateInfo(
                    milvus_pb2.GetReplicateInfoRequest(
                        source_cluster_id=source_cluster_id, target_pchannel=pch
                    ),
                    metadata=auth_metadata(observer.token),
                    timeout=10,
                )
                out[pch] = int(resp.checkpoint.time_tick)
            except grpc.RpcError:
                out[pch] = None  # unreachable, not idle
    finally:
        channel.close()
    return out


def fetch_cdc_latency(metrics_url):
    """Read the source CDC pod's /metrics endpoint directly (no Prometheus
    needed) and return {target_pchannel: avg_e2e_latency_ms}.

    True replication lag is measured *inside* CDC (it alone sees each message's
    source-produce time and target-ack time) and exported as the
    milvus_cdc_replicate_end_to_end_latency histogram (unit: milliseconds, per
    milvus source). We can't recompute it from the outside, but every milvus pod
    exposes /metrics — so we read it straight from the CDC pod.

    avg = sum/count is cumulative since the CDC pod started (not a sliding
    window). Returns None if the endpoint is unreachable.
    """
    import urllib.request
    import re
    url = metrics_url if metrics_url.rstrip("/").endswith("/metrics") else metrics_url.rstrip("/") + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            text = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    sums, counts = {}, {}
    pat = re.compile(r'target_channel_name="([^"]+)".*?\}\s+([0-9.eE+]+)\s*$')
    for line in text.splitlines():
        if not line.startswith("milvus_cdc_replicate_end_to_end_latency_"):
            continue
        m = pat.search(line)
        if not m:
            continue
        tch, val = m.group(1), float(m.group(2))
        if "_sum{" in line:
            sums[tch] = val
        elif "_count{" in line:
            counts[tch] = val
    return {t: sums[t] / counts[t] for t in sums if counts.get(t, 0) > 0}


def prefetch_salvage_checkpoints(target, source_cluster_id, pchannels):
    """Snapshot the live ReplicateCheckpoint for every target pchannel,
    while target is still in SECONDARY state and GetReplicateInfo works.

    Why this exists: see milvus-io/milvus#50344. After force_promote, the
    only client-facing API that exposes the (later-persisted) salvage_checkpoint
    is GetReplicateInfo, but its handler returns early on a standalone primary
    and never reaches the GetSalvageCheckpoint call. Until that's fixed, the
    only way to obtain a checkpoint for Data Salvage is to grab it BEFORE
    flipping the cluster's role — which is exactly what this function does.

    The live ReplicateCheckpoint at the instant just before force-promote is
    semantically the value milvus would have persisted as salvage_checkpoint;
    after the source died there are no new messages being forwarded, so live
    and persisted converge.

    Returns a list of dicts, one per pchannel:
        {target_pchannel, source_pchannel_topic, message_id_base36,
         kafka_offset, time_tick, status}
    status is "ok" / "unreachable" / "empty" / "rpc_error: ..." for triage.
    """
    stub, channel = target.stub()
    results = []
    try:
        for pch in pchannels:
            entry = {"target_pchannel": pch}
            try:
                resp = stub.GetReplicateInfo(
                    milvus_pb2.GetReplicateInfoRequest(
                        source_cluster_id=source_cluster_id, target_pchannel=pch),
                    metadata=auth_metadata(target.token),
                    timeout=10)
                cp = resp.checkpoint
                if not cp or not cp.message_id or not cp.message_id.id:
                    entry["status"] = "empty"
                else:
                    raw = cp.message_id.id
                    mid_s = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
                    entry["source_pchannel_topic"] = cp.pchannel
                    entry["message_id_base36"] = mid_s
                    try:
                        entry["kafka_offset"] = int(mid_s, 36)
                    except Exception:
                        entry["kafka_offset"] = None  # non-kafka MQ (pulsar, woodpecker)
                    entry["time_tick"] = int(cp.time_tick)
                    entry["status"] = "ok"
            except grpc.RpcError as e:
                entry["status"] = f"rpc_error: {e.code().name if hasattr(e, 'code') else 'unknown'}"
            results.append(entry)
    finally:
        channel.close()
    return results


# --------------------------------------------------------------------------- #
# milvus-backup CLI wrappers
