from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional

from pydantic import BaseModel, Field


GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")  # Should be in your .env file

# Define your data structure
class Dog(BaseModel):
    """Identifying information about a dog."""
    name: str = Field(..., description="The dog's name")
    color: str = Field(..., description="The dog's color")
    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or "gemini-1.5-pro"
    temperature=0,
    max_tokens=None,
    timeout=None,
    api_key=GOOGLE_API_KEY
)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a world class algorithm for extracting information in structured formats."),
        ("human", "Use the given format to extract information from the following input: {input}"),
        ("human", "Tip: Make sure to answer in the correct format"),
    ]
)

# Create chain
chain = prompt | llm.with_structured_output(Dog)

# Invoke chain
res = chain.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
print(res)
