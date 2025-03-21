import asyncio
from openai import OpenAI
from pydantic import BaseModel, Field
from timer import Timer
timer = Timer()
import chates


model_id = "Qwen/Qwen2.5-3B-Instruct"
client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

class ResponseFormat(BaseModel):
    name : str
    age: int
    

inputs = [
        "  \"I\'m vishva, I'm 26 years old '\"",
        "  \"I\'m Vimal, I'm 45 years old '\"",
        "  \"I\'m Ragul, I'm 31 years old '\""
    ]
    
    
async def process_conversation(conversation):
    prompt = """
    Extract the following answers for the questions from the above conversation:
    give me name and age of the person as json.
    """
    old_sys_prompt = "You are a information extractor. Return structured JSON string only."
    completion = await client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a information extractor. Return structured JSON string only."},
            {"role": "user", "content": conversation + prompt},
        ],
        extra_body={"guided_json": ResponseFormat.model_json_schema()},
        response_format="json"  # Request JSON response
    )
    print(completion.choices[0].message.parsed)

async def main():
    tasks = [process_conversation(text) for text in inputs]
    await asyncio.gather(*tasks)


timer.start()
asyncio.run(main())
timer.stop()
