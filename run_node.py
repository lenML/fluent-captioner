import base64
import io
import logging
import os
import time
import uuid

import fastapi_jsonrpc as jsonrpc
import numpy as np
from fastapi import Body
from PIL import Image
from pydantic import BaseModel

from modules.infer_node import HealthCheckModel, NodeInfoModel, model_infer_node
from modules.models.Florence2 import (
    Florence2Model,
    OCRRegionResult,
    ODResult,
    SEGResult,
    TextResult,
)
from modules.TaskQueue import TaskQueue

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

app = jsonrpc.API()

api_v1 = jsonrpc.Entrypoint("/v1/jsonrpc")

model = Florence2Model()


def load_base64_image(b64: str):
    if b64.startswith("data:image"):
        b64 = b64.split(",")[1]
    base64_decoded = base64.b64decode(b64)
    image = Image.open(io.BytesIO(base64_decoded))
    image_np = np.array(image)
    return image_np


def process_image_work(
    task_queue: TaskQueue,
    task_id: str,
    # -
    image_b64: str,
    task_type: str,
    text_input: str,
):
    start_time = time.time()
    logger.info(f"[Task:{task_id}] started")

    model.load_model()
    image = load_base64_image(image_b64)
    result = model.run_task(
        image=image,
        task_type=task_type,
        text_input=text_input,
    )

    logger.info(f"[Task:{task_id}] completed in {time.time() - start_time:.2f}s")

    return result


tq = TaskQueue(process_task_func=process_image_work)


@api_v1.method()
def get_node_info() -> NodeInfoModel:
    return model_infer_node.get_info()


@api_v1.method()
def health_check() -> HealthCheckModel:
    return model_infer_node.health_check()


class InferResponse(BaseModel):
    text: None | str = None
    ocr_region: None | OCRRegionResult = None
    od: None | ODResult = None
    seg: None | SEGResult = None


class TaskStatus(BaseModel):
    task_id: str
    status: str
    result: InferResponse | None = None
    error: str | None


@api_v1.method()
def get_task_status(task_id: str) -> TaskStatus:
    task = tq.get_task_status(task_id)
    if task is None:
        raise RpcError(data=RpcError.DataModel(details="Task not found"))
    if task.status == "completed":
        task_result = task.result
        result = InferResponse()
        if isinstance(task_result, TextResult):
            result.text = task_result.text
        elif isinstance(task_result, OCRRegionResult):
            result.ocr_region = task_result
        elif isinstance(task_result, ODResult):
            result.od = task_result
        elif isinstance(task_result, SEGResult):
            result.seg = task_result
        else:
            raise RpcError(data=RpcError.DataModel(details="Unknown result type"))
    else:
        result = None
    return TaskStatus(
        task_id=task_id,
        status=task.status,
        result=result,
        error=task.error,
    )


class RpcError(jsonrpc.BaseError):
    CODE = 5000
    MESSAGE = "RPC Error"

    _log_id: str = uuid.uuid4().hex

    def __init__(self, data):
        super().__init__(data=data)
        logger.error(f"[RPC Error] {data}")

    class DataModel(BaseModel):
        details: str


@api_v1.method()
def model_inference(
    model_name: str = Body(..., examples=["florence-2"]),
    params: dict = Body(...),
) -> TaskStatus:
    if model_name == "florence-2":
        image_b64 = params.get("image")
        task_type = params.get("task_type")
        text_input = params.get("text_input")
        if image_b64 is None:
            raise RpcError(data=RpcError.DataModel(details="params.image is required"))
        if task_type is None:
            raise RpcError(
                data=RpcError.DataModel(details="params.task_type is required")
            )

        task_id = tq.add_task(
            image_b64=image_b64,
            task_type=task_type,
            text_input=text_input,
        )
        tq.start_worker()

        return get_task_status(task_id)
    else:
        raise RpcError(data=RpcError.DataModel(details="Model not found"))


app.bind_entrypoint(api_v1)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
