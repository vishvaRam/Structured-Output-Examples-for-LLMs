import json
from typing import List, Optional

import outlines
import torch
from PIL import Image
from outlines import samplers
from pydantic import BaseModel, Field
from rich import print
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration


class ImageDescription(BaseModel):
    """
    Pydantic class to represent an image description.
    """
    subject: str = Field(description="The main subject of the image.")
    action: Optional[str] = Field(None, description="The action being performed in the image, if any.")
    objects: Optional[List[str]] = Field(None, description="A list of objects present in the image.")
    scene: Optional[str] = Field(None, description="The general scene or setting of the image.")
    setting: Optional[str] = Field(None, description="Specific details of the setting or environment.")
    colors: Optional[List[str]] = Field(None, description="Dominant colors present in the image.")
    style: Optional[str] = Field(None, description="The artistic style or photographic technique used.")
    mood: Optional[str] = Field(None, description="The overall mood or atmosphere of the image.")
    composition: Optional[str] = Field(None, description="The composition of the image (e.g., close-up, wide shot).")
    lighting: Optional[str] = Field(None, description="The lighting conditions in the image.")
    details: Optional[List[str]] = Field(None, description="Specific details or notable features in the image.")
    additional_notes: Optional[str] = Field(None, description="Any additional relevant information about the image.")


def load_and_resize_image(image_path, max_size=1024):
    """
    Load and resize an image while maintaining aspect ratio

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) of the output image

    Returns:
        PIL Image: Resized image
    """
    image = Image.open(image_path)

    # Get current dimensions
    width, height = image.size

    # Calculate scaling factor
    scale = min(max_size / width, max_size / height)

    # Only resize if image is larger than max_size
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def save_resized_image(resized_image, output_path, format="PNG", quality=95):
    """
    Save a PIL Image object to a specified path.

    Args:
        resized_image: The PIL Image object to save.
        output_path: The path where the image should be saved.
        format: The image format to save as (e.g., "PNG", "JPEG"). Defaults to "PNG".
        quality: The quality for JPEG images (0-95, higher is better). Ignored for other formats.
    """
    try:
        resized_image.save(output_path, format=format, quality=quality)
        print(f"Resized image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")


resized_img = load_and_resize_image(image_path="Inputs/for_enhance.jpg")

resized = "resized.jpg"
save_resized_image(resized_image=resized_img, quality=100, format="JPEG", output_path=resized)

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model_class = Qwen2_5_VLForConditionalGeneration
model = outlines.models.transformers_vision(
    model_name,
    model_class=model_class,
    model_kwargs={
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "do_sample": True
    }
)

# Corrected messages structure for batch processing
messages = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": resized_img,
                },
                {
                    "type": "text",
                    "text": f"""You are an expert at generating detailed image descriptions.
                    Please provide a comprehensive description of the image. Be as detailed as possible.
                    Return the information in the following JSON schema:
                    {ImageDescription.model_json_schema()}
                    """,
                },
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": resized_img,
                },
                {
                    "type": "text",
                    "text": f"""You are an expert at generating detailed image descriptions.
                    Please provide a comprehensive description of the image. Be as detailed as possible.
                    Return the information in the following JSON schema:
                    {ImageDescription.model_json_schema()}
                    """,
                },
            ],
        }
    ],
]

# Convert the messages to the final prompt
processor = AutoProcessor.from_pretrained(model_name)
prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

# print(prompts)

image_description_generator = outlines.generate.json(
    model,
    ImageDescription,
    sampler=samplers.GreedySampler()
)

# Generate the image description for each prompt and image
# pass the images within the prompts.
results = image_description_generator(prompts, [[resized_img],[resized_img]])

# Print the results
for result in results:
    print(json.dumps(result.model_dump(), indent=4))
