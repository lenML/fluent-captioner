import logging
import os
from typing import Literal, Tuple, Union

# workaround for unnecessary flash_attn requirement
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    ValidationError,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from transformers import AutoModelForCausalLM, AutoProcessor, CLIPImageProcessor
from transformers.dynamic_module_utils import get_imports

logger = logging.getLogger(__name__)


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


task_mapping = {
    "Caption": "<CAPTION>",
    "Detailed Caption": "<DETAILED_CAPTION>",
    "More Detailed Caption": "<MORE_DETAILED_CAPTION>",
    "Object Detection": "<OD>",
    "Dense Region Caption": "<DENSE_REGION_CAPTION>",
    "Region Proposal": "<REGION_PROPOSAL>",
    "Caption to Phrase Grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "Referring Expression Segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "Region to Segmentation": "<REGION_TO_SEGMENTATION>",
    "Open Vocabulary Detection": "<OPEN_VOCABULARY_DETECTION>",
    "Region to Category": "<REGION_TO_CATEGORY>",
    "Region to Description": "<REGION_TO_DESCRIPTION>",
    "OCR": "<OCR>",
    "OCR with Region": "<OCR_WITH_REGION>",
}
TASK_TYPES = Literal[
    "Caption",
    "Detailed Caption",
    "More Detailed Caption",
    "Object Detection",
    "Dense Region Caption",
    "Region Proposal",
    "Caption to Phrase Grounding",
    "Referring Expression Segmentation",
    "Region to Segmentation",
    "Open Vocabulary Detection",
    "Region to Category",
    "Region to Description",
    "OCR",
    "OCR with Region",
]


class ODResult(BaseModel):
    bbox: list[Tuple[int, int, int, int]]
    labels: list[str]


class OCRRegionResult(BaseModel):
    quad_boxes: list[Tuple[int, int, int, int, int, int, int, int]]
    labels: list[str]


class SEGResult(BaseModel):
    polygons: list[list[list[int]]]
    labels: list[str]


class TextResult(BaseModel):
    text: str


class Florence2Model:
    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large",
        device=torch.device("cuda"),
        # 使用 f16 会有明显质量下降...
        dtype=torch.float16,
        max_new_tokens=2048,
        num_beams=3,
    ) -> None:
        assert max_new_tokens > 0, "max_new_tokens should be greater than 0"
        assert num_beams > 0, "num_beams should be greater than 0"

        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        self.model = None
        self.processor: CLIPImageProcessor = None

    def load_model(self):
        if self.model:
            return

        logger.info(f"Loading model: {self.model_id}")

        with patch(
            "transformers.dynamic_module_utils.get_imports", fixed_get_imports
        ):  # workaround for unnecessary flash_attn requirement
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    cache_dir="./models",
                    trust_remote_code=True,
                    local_files_only=True,
                )
                .to(device=self.device, dtype=self.dtype)
                .eval()
            )
            self.processor: CLIPImageProcessor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir="./models",
                trust_remote_code=True,
                local_files_only=True,
            )

        logger.info(f"Model loaded: {self.model_id}")

    def process_image(
        self,
        image: np.ndarray,
        task_type: TASK_TYPES,
        text_input: Union[str, None] = None,
    ) -> Union[str, dict]:
        image = Image.fromarray(image)
        if task_type not in task_mapping:
            raise ValueError(f"Invalid task type [{task_type}]")

        task_prompt = task_mapping[task_type]
        parsed_answer = self.run_model(task_prompt, image, text_input)
        return parsed_answer

    def run_model(
        self, task_prompt: str, image: Image.Image, text_input: Union[str, None] = None
    ) -> str:
        assert self.model, "Model is not loaded"

        if text_input is not None:
            prompt = task_prompt + text_input
        else:
            prompt = task_prompt

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            device=self.device, dtype=self.dtype
        )

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            early_stopping=False,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # fix: REGION_TO_CATEGORY => OD format
        if task_prompt == "<REGION_TO_CATEGORY>":
            task_prompt = "<OD>"

        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        return parsed_answer

    def run_task(
        self,
        image: np.ndarray,
        task_type: TASK_TYPES,
        text_input: Union[str, None] = None,
    ):
        parsed_answer = self.process_image(
            image=image, task_type=task_type, text_input=text_input
        )
        result = parsed_answer[list(parsed_answer.keys())[0]]
        if isinstance(result, str):
            return TextResult(text=result)
        if "bboxes" in result:
            return ODResult(bboxes=result["bboxes"], labels=result["labels"])
        if "quad_boxes" in result:
            return OCRRegionResult(
                quad_boxes=result["quad_boxes"], labels=result["labels"]
            )
        if "polygons" in result:
            return SEGResult(polygons=result["polygons"], labels=result["labels"])
        raise ValueError(f"Invalid result format: {result}")

    def run_caption_task(
        self,
        image: np.ndarray,
        task_type: Literal[
            "Caption", "Detailed Caption", "More Detailed Caption"
        ] = "More Detailed Caption",
    ) -> TextResult:
        return self.run_task(image, task_type)

    def run_ocr_task(
        self,
        image: np.ndarray,
    ) -> TextResult:
        return self.run_task(image, task_type="OCR")

    def run_region_ocr_task(
        self,
        image: np.ndarray,
    ) -> OCRRegionResult:
        return self.run_task(image, task_type="OCR with Region")

    def run_od_task(
        self,
        image: np.ndarray,
    ) -> ODResult:
        return self.run_task(image, task_type="Object Detection")

    def run_open_od_task(
        self,
        image: np.ndarray,
        prompt: str,
    ) -> ODResult:
        return self.run_task(
            image, task_type="Open Vocabulary Detection", text_input=prompt
        )

    def run_region_caption_task(
        self,
        image: np.ndarray,
    ) -> ODResult:
        return self.run_task(image, task_type="Dense Region Caption")

    def run_region_proposal_task(
        self,
        image: np.ndarray,
    ) -> ODResult:
        return self.run_task(image, task_type="Region Proposal")

    def run_grounding_task(
        self,
        image: np.ndarray,
        caption: str,
    ) -> ODResult:
        return self.run_task(
            image, task_type="Caption to Phrase Grounding", text_input=caption
        )

    def run_segmentation_task(
        self,
        image: np.ndarray,
        ref_prompt: str,
    ) -> SEGResult:
        return self.run_task(
            image, task_type="Referring Expression Segmentation", text_input=ref_prompt
        )

    def run_region_segmentation_task(
        self,
        image: np.ndarray,
        regin_prompt: str,
    ) -> SEGResult:
        return self.run_task(
            image, task_type="Region to Segmentation", text_input=regin_prompt
        )

    def run_region_category_task(
        self,
        image: np.ndarray,
        region_prompt: str,
    ) -> ODResult:
        return self.run_task(
            image, task_type="Region to Category", text_input=region_prompt
        )

    def run_region_description_task(
        self,
        image: np.ndarray,
        region_prompt: str,
    ) -> TextResult:
        return self.run_task(
            image, task_type="Region to Description", text_input=region_prompt
        )


if __name__ == "__main__":
    # model = Florence2Model(dtype=torch.float32)
    model = Florence2Model()
    model.load_model()
    # image = np.array(Image.open("./test.png"))
    image = Image.open("./test_ocr.png")
    image = image.convert("RGB")
    image_data = np.array(image)
    # result = model.process_image(image, task_type="Region Proposal")
    result = model.run_region_category_task(image_data, "logo")
    print("== OUTPUT ==")
    print(result)
