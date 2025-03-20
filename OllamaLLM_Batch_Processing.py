from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class ResponseFormat(BaseModel):
    name : str
    age: int
    
parser = PydanticOutputParser(pydantic_object=ResponseFormat)

model = OllamaLLM(
        model="qwen2.5:0.5b",
        format="json"
    )

# Combine model with parser for automatic post-processing
chain = model | parser

# Await the abatch call
result = chain.batch([
        "give me name and age of the person as json. \"I\'m vishva, I'm 26 years old '\"",
        "give me name and age of the person as json. \"I\'m Vimal, I'm 45 years old '\"",
        "give me name and age of the person as json. \"I\'m Ragul, I'm 31 years old '\""
    ])

print(result)