import os
from openai import OpenAI
from typing import Optional

from pydantic import BaseModel, Field

class Dog(BaseModel):
    """Identifying information about a dog."""
    name: str = Field(..., description="The dog's name")
    color: str = Field(..., description="The dog's color")
    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# For Gemini
client = OpenAI(
                api_key=os.getenv("GOOGLE_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
# For Groq
client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY")
            )

# For Local
client = OpenAI(
    base_url="http://ollama/v1",
    api_key="not-needed"
)

input_text="Harry was a chubby brown beagle who loved chicken"

response = client.chat.completions.parse(
        model="gpt-3.5-turbo",  #  Not necessary of not using openAI
        messages=[
            {"role": "system", "content": "You are a world-class algorithm for extracting information in structured formats. Extract the relevant information about a dog from the user's input."},
            {"role": "user", "content": input_text}
        ],
        response_format=Dog,  # Specify the Pydantic model for structured output
        temperature=0.1,
        max_tokens=512,
    )

print(response.choices[0].message.parsed)

# output
# name='Harry' color='brown' fav_food='chicken'
