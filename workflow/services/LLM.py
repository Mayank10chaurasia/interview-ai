# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# import os

# load_dotenv()

# api_key = os.getenv("OPENROUTER_API_KEY")

# llm = ChatOpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=api_key,
#     model="meta-llama/llama-3.1-8b-instruct",  # change model
# )

# print("Loaded:", api_key[:15] + "...")

# try:
#     response = llm.invoke("Say hello.")
#     print(response.content)
# except Exception as e:
#     print(type(e).__name__)
#     print(e)

from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # or mixtral-8x7b-32768, gemma2-9b-it
    api_key="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"     # free at console.groq.com

)


from langchain_ollama import OllamaLLM

llm1 = OllamaLLM(
    model="phi3:latest",
    temperature=0
)