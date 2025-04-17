from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variables
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

result = client.chat.completions.create(
    model='gemini-1.5-flash-8b-001',
    messages=[
        {'role': 'system', 'content': 'You are an ai assistant whose name is ChaiCode'},
        {'role': 'user', 'content': 'Hey there, what is your name? I am Talha'} # system prompt
    ]
)

print(result.choices[0].message.content)
