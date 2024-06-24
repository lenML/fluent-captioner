import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, CLIPImageProcessor
from PIL import Image
import numpy as np
from typing import Tuple, Union, Literal

# workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports


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


class ODResult:
    def __init__(self, bboxes: list[Tuple[int, int, int, int]], labels: list[str]):
        self.bboxes = bboxes
        self.labels = labels

        self.objects = list(zip(self.bboxes, self.labels))

    def __str__(self):
        return str(self.objects)


class OCRRegionResult:
    def __init__(
        self,
        quad_boxes: list[Tuple[int, int, int, int, int, int, int, int]],
        labels: list[str],
    ):
        self.quad_boxes = quad_boxes
        self.labels = labels

        self.objects = list(zip(self.quad_boxes, self.labels))

    def __str__(self):
        return str(self.objects)


class SEGResult:
    def __init__(self, polygons: list[list[list[int]]], labels: list[str]):
        self.polygons = polygons
        self.labels = labels

        self.objects = list(zip(self.polygons, self.labels))

    def __str__(self):
        return str(self.objects)


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

        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        with patch(
            "transformers.dynamic_module_utils.get_imports", fixed_get_imports
        ):  # workaround for unnecessary flash_attn requirement
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model_id,
                    cache_dir="./models",
                    trust_remote_code=True,
                    local_files_only=True,
                )
                .to(device=self.device, dtype=dtype)
                .eval()
            )
            self.processor: CLIPImageProcessor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir="./models",
                trust_remote_code=True,
                local_files_only=True,
            )

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
        return result

    def run_caption_task(
        self,
        image: np.ndarray,
        task_type: Literal[
            "Caption", "Detailed Caption", "More Detailed Caption"
        ] = "More Detailed Caption",
    ) -> str:
        return self.run_task(image, task_type)

    def run_ocr_task(
        self,
        image: np.ndarray,
    ) -> str:
        return self.run_task(image, task_type="OCR")

    def run_region_ocr_task(
        self,
        image: np.ndarray,
    ) -> OCRRegionResult:
        result = self.run_task(image, task_type="OCR with Region")
        return OCRRegionResult(quad_boxes=result["quad_boxes"], labels=result["labels"])

    def run_od_task(
        self,
        image: np.ndarray,
    ) -> ODResult:
        result = self.run_task(image, task_type="Object Detection")
        return ODResult(bboxes=result["bboxes"], labels=result["labels"])

    def run_open_od_task(
        self,
        image: np.ndarray,
        prompt: str,
    ) -> ODResult:
        result = self.run_task(
            image, task_type="Open Vocabulary Detection", text_input=prompt
        )
        return ODResult(bboxes=result["bboxes"], labels=result["labels"])

    def run_region_caption_task(
        self,
        image: np.ndarray,
    ) -> ODResult:
        result = self.run_task(image, task_type="Dense Region Caption")
        return ODResult(bboxes=result["bboxes"], labels=result["labels"])

    def run_region_proposal_task(
        self,
        image: np.ndarray,
    ) -> ODResult:
        result = self.run_task(image, task_type="Region Proposal")
        return ODResult(bboxes=result["bboxes"], labels=result["labels"])

    def run_grounding_task(
        self,
        image: np.ndarray,
        caption: str,
    ) -> ODResult:
        result = self.run_task(
            image, task_type="Caption to Phrase Grounding", text_input=caption
        )
        return ODResult(bboxes=result["bboxes"], labels=result["labels"])

    def run_segmentation_task(
        self,
        image: np.ndarray,
        ref_prompt: str,
    ) -> SEGResult:
        result = self.run_task(
            image, task_type="Referring Expression Segmentation", text_input=ref_prompt
        )
        return SEGResult(polygons=result["polygons"], labels=result["labels"])

    def run_region_segmentation_task(
        self,
        image: np.ndarray,
        regin_prompt: str,
    ) -> SEGResult:
        result = self.run_task(
            image, task_type="Region to Segmentation", text_input=regin_prompt
        )
        return SEGResult(polygons=result["polygons"], labels=result["labels"])

    def run_region_category_task(
        self,
        image: np.ndarray,
        region_prompt: str,
    ) -> ODResult:
        result = self.run_task(
            image, task_type="Region to Category", text_input=region_prompt
        )
        return ODResult(bboxes=result["bboxes"], labels=result["labels"])

    def run_region_description_task(
        self,
        image: np.ndarray,
        region_prompt: str,
    ) -> str:
        return self.run_task(
            image, task_type="Region to Description", text_input=region_prompt
        )


if __name__ == "__main__":
    # model = Florence2Model(dtype=torch.float32)
    model = Florence2Model()
    # image = np.array(Image.open("./test.png"))
    image = Image.open("./test_ocr.png")
    image = image.convert("RGB")
    image_data = np.array(image)
    # result = model.process_image(image, task_type="Region Proposal")
    result = model.run_region_category_task(image_data, "logo")
    print("== OUTPUT ==")
    print(result)
