"""Cluster handle + low-level helpers (grpc addr, pchannels, auth, status)."""
import base64
from urllib.parse import urlparse

import grpc
from pymilvus.grpc_gen import common_pb2, milvus_pb2_grpc


def grpc_addr(uri):
    if "://" in uri:
        parsed = urlparse(uri)
        return f"{parsed.hostname or '127.0.0.1'}:{parsed.port or 19530}"
    return uri


def pchannels_of(cluster_id, num):
    return [f"{cluster_id}-rootcoord-dml_{i}" for i in range(num)]


def auth_metadata(token):
    if not token:
        return []
    return [("authorization", base64.b64encode(token.encode()).decode())]


def status_ok(status):
    code = getattr(status, "code", 0) or 0
    err = getattr(status, "error_code", 0) or 0
    return code == 0 and err in (0, common_pb2.ErrorCode.Success)



# --------------------------------------------------------------------------- #
class Cluster:
    def __init__(self, role, uri, cluster_id, pchannel_num, token, inter_uri=None,
                 grpc_override=None, cdc_metrics=None):
        self.role = role
        self.uri = uri
        self.cluster_id = cluster_id
        self.pchannel_num = pchannel_num
        self.token = token
        self.dial_addr = grpc_override or grpc_addr(uri)
        self.inter_uri = inter_uri or grpc_addr(uri)
        self.cdc_metrics = cdc_metrics  # source CDC pod /metrics endpoint (optional)

    def milvus_cluster(self):
        return common_pb2.MilvusCluster(
            cluster_id=self.cluster_id,
            connection_param=common_pb2.ConnectionParam(uri=self.inter_uri, token=self.token),
            pchannels=pchannels_of(self.cluster_id, self.pchannel_num),
        )

    def stub(self):
        channel = grpc.insecure_channel(self.dial_addr)
        return milvus_pb2_grpc.MilvusServiceStub(channel), channel


