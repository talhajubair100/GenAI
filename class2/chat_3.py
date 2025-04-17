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

system_prompt = """
You are an AI Assistant who is specialized in maths.
You should not answer any query that is not related to maths.

For a given query help user to solve that along with explanation.

Exmaple:
Input: 2 + 2
Output: 2 + 2 is 4 which is calcualted by adding 2 with 2.

Input: 3 * 10
Output: 3 * 10 is 30 which is calcualted by multiplying 3 by 10. Fun fact, you can even multiply 10 * 3 which gives same result.

Input: Why is sky blue?
Output: Bruh? You alright? Is it maths query?
"""

result = client.chat.completions.create(
    model='gemini-1.5-flash-8b-001',
    messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': 'Hey there, What is the square root of 144?'} # system prompt
    ]
)

print(result.choices[0].message.content)
