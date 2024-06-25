import uuid

from pydantic import BaseModel


class NodeInfoModel(BaseModel):
    version: str
    status: str
    queue_length: int
    pending_jobs: list
    id: str


class HealthCheckModel(BaseModel):
    ok: bool
    # TODO gpu


class ModelInferNode:
    def __init__(self):
        self.version = "0.1"
        self.status = "idle"
        self.queue_length = 0
        self.pending_jobs = []
        self.id = uuid.uuid4()

    def health_check(self):
        return HealthCheckModel(ok=True)

    def get_info(self):
        return NodeInfoModel(
            version=self.version,
            status=self.status,
            queue_length=self.queue_length,
            pending_jobs=self.pending_jobs,
            id=str(self.id),
        )


model_infer_node = ModelInferNode()
