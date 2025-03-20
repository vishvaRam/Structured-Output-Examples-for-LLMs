from ollama import chat
from pydantic import BaseModel

class ResponseFormat(BaseModel):
    name : str
    age: int
    
prompt ="""give me name and age of the person as json. 
            \"I\'m vishva, I'm 26 years old '\"" 
        """

# Structured output from Vision LLM

response = chat(
    messages=[{
        'role': 'user',
        'content': prompt,
        'images': ["image.jpeg"]
    }],
    model='minicpm-v',
    format=ResponseFormat.model_json_schema(),
)