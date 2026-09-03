

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("openAI_key")

llm = ChatOpenAI(
    model="qwen3.6",
    base_url="https://ai.tcetcercd.in/v1",
    api_key=api_key,
    max_tokens=500
)
