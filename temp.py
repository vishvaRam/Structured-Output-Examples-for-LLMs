import json
import outlines
import torch
from typing import List, Optional
from pydantic import BaseModel, Field

if torch.cuda.is_available():
    print("GPU is Used!")
else:
    print("CPU is Used!")

repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
model = outlines.models.transformers(repo_id,
                                     device="cuda",
                                     model_kwargs={"temperature":0.5})

input_list =[
    "My cat, Whiskers, enjoys a variety of toys: feather wands, laser pointers, and those little crinkly balls",
    "Spot, our energetic dog, loves his snacks: peanut butter biscuits, chewy ropes, and the occasional carrot stick.",
    ]

class PetInfo(BaseModel):
    pet_name: str = Field(..., description="The name of the pet.")
    pet_type: Optional[str] = Field(None, description="The type of the pet (e.g., cat, dog).")
    items: List[str] = Field(..., description="List of items the pet enjoys.")

schema_as_str = json.dumps(PetInfo.model_json_schema())
generator = outlines.generate.json(model, schema_as_str)

output = generator(input_list)
print(json.dumps(output,indent=4))