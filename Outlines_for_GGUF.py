import json
import outlines
import torch
from llama_cpp import Llama
from outlines import models, generate
from pydantic import BaseModel, Field

class ResponseFormat(BaseModel):
    name: str
    age: int

if torch.cuda.is_available():
    print("GPU is Used!")
else:
    print("CPU is Used!")

model_path = "./Meta-Llama-3.1-8B-Instruct-GGUF.gguf"

# Initialize the GGUF model
llm = Llama(
    model_path=model_path,
    n_ctx=512,  # Context window
    n_gpu_layers=-1 if torch.cuda.is_available() else 0,  # Use GPU if available
    verbose=False
)

# Create the outlines model
model = models.LlamaCpp(llm)

# Create structured generator using the Pydantic model
generator = generate.json(model, ResponseFormat)

inputs = "Give me the name and age of the person: \"I'm Vishva, I'm 26 years old\""

# Generate structured response
response = generator(inputs)

print("Generated response:")
print(response)
print(f"Type: {type(response)}")

# If you want to access individual fields
if isinstance(response, ResponseFormat):
    print(f"Name: {response.name}")
    print(f"Age: {response.age}")


# Output:
# GPU is Used!
# llama_context: n_ctx_per_seq (512) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
# Generated response:
# name='Vishva' age=26
# Type: <class '__main__.ResponseFormat'>
# Name: Vishva
# Age: 26
