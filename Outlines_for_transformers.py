import json
import outlines
import torch
from typing import List, Optional
from pydantic import BaseModel, Field

class ResponseFormat(BaseModel):
    name : str
    age: int

if torch.cuda.is_available():
    print("GPU is Used!")
else:
    print("CPU is Used!")

repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
model = outlines.models.transformers(repo_id,
                                     device="cuda",
                                     model_kwargs={"temperature":0.5})

inputs = [
        "give me name and age of the person as json. \"I\'m vishva, I'm 26 years old '\"",
        "give me name and age of the person as json. \"I\'m Vimal, I'm 45 years old '\"",
        "give me name and age of the person as json. \"I\'m Ragul, I'm 31 years old '\""
    ]
  
schema_as_str = json.dumps(ResponseFormat.model_json_schema())
generator = outlines.generate.json(model, schema_as_str)

output = generator(inputs)
print(json.dumps(output,indent=4))
