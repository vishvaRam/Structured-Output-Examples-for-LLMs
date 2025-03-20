from langchain_ollama import ChatOllama
from pydantic import BaseModel

class ResponseFormat(BaseModel):
    name : str
    age: int

model = ChatOllama(
        model="qwen2.5:0.5b",
        format=ResponseFormat.model_json_schema()
    )

out = model.invoke("give me name and age of the person as json. \"I\'m vishva, I'm 26 years old '\"")
print(out.content)
