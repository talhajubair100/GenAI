import json
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
You are an AI assistant who is expert in breaking down complex problems and then resolve the user query

For the given user input, analyse the input and break down the problem step by step.
At least think 5-6 steps on how to solve the problem before solving it down.

The steps are you get a user input, you analyse, you think, you again think for several times and then return an output with explanation and then finally you validate the output as well before giving final result.

Follow the steps in sequence that is "analyse", "think", "output", "validate", and finally "result".

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query

Output Format:
{{ step: "string", content: "string" }}

Example:
Input: What is 2 + 2
Output: {{ step: "analyse", content: "Alright! The user is interested in maths query and he is asking a basic athematic operation" }}
Output: {{ step: "think", content: "To perform the addition i must go from left to right and add all the operands" }}
Output: {{ step: "output", content: "4" }}
Output: {{ step: "validate", content: "seems like 4 is correct answer for 2 + 2" }}
Output: {{ step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers" }}

"""

messages = [
    {'role': 'system', 'content': system_prompt},
]

query = input("Enter your query: ")
messages.append({'role': 'user', 'content': query}) # system prompt

while True:
    result = client.chat.completions.create(
        model='gemini-1.5-flash-8b-001',
        response_format={"type": "json_object"},
        messages=messages
    )
    try:
        # Attempt to parse the response as JSON
        parsed_response = json.loads(result.choices[0].message.content)
        messages.append({"role": "assistant", "content": json.dumps(parsed_response)})
        
        if parsed_response.get("step") != "output":
            print("parsed_response: ", parsed_response.get("content"))
            continue 
        
        print("final result --->", parsed_response.get("content"))
        break
    except json.JSONDecodeError as e:
        # Handle JSON parsing errors
        print("Error parsing response as JSON:", e)
        print("Raw response content:", result.choices[0].message.content)
        break

