from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional

from pydantic import BaseModel, Field

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class Dog(BaseModel):
    """Identifying information about a dog."""

    name: str = Field(..., description="The dog's name")
    color: str = Field(..., description="The dog's color")
    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    groq_api_key=GROQ_API_KEY,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a world class algorithm for extracting information in structured formats."),
        ("human", "Use the given format to extract information from the following input: {input}"),
        ("human", "Tip: Make sure to answer in the correct format"),
    ]
)

chain = prompt | llm.with_structured_output(Dog)

res = chain.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
print(res)

# Output -----
# name='Harry' color='brown' fav_food='chicken'

