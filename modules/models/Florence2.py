from transformers import AutoProcessor, AutoModelForCausalLM, CLIPImageProcessor
from PIL import Image
import numpy as np
from typing import Tuple, Union, Literal


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


class Florence2Model:
    def __init__(self, model_id: str = "microsoft/Florence-2-large") -> None:
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir="./models", trust_remote_code=True
            )
            .to("cuda")
            .eval()
        )
        self.processor: CLIPImageProcessor = AutoProcessor.from_pretrained(
            model_id, cache_dir="./models", trust_remote_code=True
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
        result = self.run_model(task_prompt, image, text_input)
        return self.parse_result(result)

    def run_model(
        self, task_prompt: str, image: Image.Image, text_input: Union[str, None] = None
    ) -> str:
        if text_input is not None:
            prompt = task_prompt + text_input
        else:
            prompt = task_prompt

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            "cuda"
        )
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        return generated_text

    def parse_result(self, result: str) -> Union[str, dict]:
        # TODO: parser
        return {"raw": result}


if __name__ == "__main__":
    model = Florence2Model()
    image = np.array(Image.open("./test.png"))
    result = model.process_image(image, task_type="More Detailed Caption")
    print(result)
