import base64
import time
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    title: str = Field(description="Title of the image")
    description: str = Field(description="Description of the image content")


# Initialize the output parser with the Pydantic model
parser = PydanticOutputParser(pydantic_object=ImageInfo)

# Load your image and encode it as base64
with open("Data/img1.jpg", "rb") as image_file:
    image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

# Initialize the Ollama LLM with the vision model llama4:scout
llm = OllamaLLM(model="llama4:scout",
                temperature=0.01,
                num_ctx=512,
                format="json")

# Bind the image context to the model
llm_with_image = llm.bind(images=[image_b64])

# Combine the model with the parser for automatic structured output parsing
chain = llm_with_image | parser

# Define the prompt for extracting title and description
prompt = (
    "Analyze the image and extract a concise title and a detailed description of its content. "
    "Return the output as JSON with the following keys: 'title', 'description'."
)

# Start the timer
start_time = time.time()

try:
    # Invoke the chain with the prompt
    result = chain.invoke(prompt)
    print("Extracted Data:", result)
except Exception as e:
    print("An error occurred:", e)

# Stop the timer
end_time = time.time()

# Calculate and print the total execution time
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")
