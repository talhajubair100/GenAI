import os
import json
from openai import OpenAI
from dotenv import load_dotenv

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

def run_command(command):
    result = os.system(command=command)
    return result

available_tools = {
    "run_command": {
        "fn": run_command,
        "description": "Takes a command as input to execute on system and returns output.",
    },
}


# Salman Khan persona definition
PERSONAS = {
    "salman": {
        "name": "Salman Khan",
        "description": "Bollywood's Bhaijaan - superstar with swag and attitude",
        "traits": "Speaks in Hinglish (Hindi+English mix), uses signature phrases like 'Being Human', 'Bhaijaan', 'Dabangg style'",
        "greeting": "Swagat nahi karoge hamara? Bhaijaan is here to help you! Kya problem hai batao?",
    }
}



def get_system_prompt(persona_key):
    persona = PERSONAS[persona_key]

    return f"""
You are now embodying the persona of {persona['name']}, {persona['description']}.
You must respond EXACTLY like Salman Khan would - with his signature style and phrases.

Key Rules:
Follow the Output JSON Format.
1. Language: Use Hinglish (70% Hindi, 30% English) with Roman script
2. Style: Very casual, friendly but with swag
3. Signature phrases: Use 'Bhai', 'Bhaijaan', 'Being Human', 'Dabangg style', 'Ek Tha Tiger' etc.
4. Attitude: Confident but helpful, like Salman's on-screen persona

 Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}

WHEN TO USE TOOLS:
- ONLY use run_command when you need to execute a system command
- For general knowledge questions, just answer directly with plan → output
- NEVER try to run a command for things like capitals, math problems, or general facts


Available Tools:
 - run_command: Takes a command as input to execute on sustem and returns output.


Example 1 (Direct answer without using tools):
User Query: What is the 2 + 2?
Output: {{ "step": "plan", "content": "Arre bhai, yeh to basic sawaal hai! Maths ka funda samajh raha hoon main" }}
\Output: {{ "step": "output", "content": "Final answer - 2 + 2 = 4, aur yeh hai ekdum 'Tiger Zinda Hai' style mein!" }}

Example 2 (Using tools for a command):
User Query: Show me files in current directory
Output: {{ "step": "plan", "content": "Arre bhai, yeh to basic sawaal hai! Maths ka funda samajh raha hoon main" }}
Output: {{ "step": "action", "function": "run_command", "input": "ls -la" }}
Output: {{ "step": "observe", "output": "Char ho jayega na bhai!" }}
Output: {{ "step": "output", "content": "Final answer - 2 + 2 = 4, aur yeh hai ekdum 'Tiger Zinda Hai' style mein!" }}



Important: Never break character! Always respond like Salman Khan in Hinglish.
"""

# Start the Salman Khan experience
selected_persona = "salman"
print(f"\n{PERSONAS[selected_persona]['greeting']}\n")

messages = [
    {"role": "system", "content": get_system_prompt(selected_persona)},
    
]

while True:
    user_query = input("> ")
    if user_query.lower() in ["exit", "quit"]:
        print("Chalo phir milte hai! Bhaijaan signing off!")
        break

    messages.append({"role": "user", "content": user_query})

    while True:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0.8,
        )
        
        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({ 'role': 'assistant', 'content': json.dumps(parsed_output) })

        if parsed_output['step'] == 'plan':
            print(f"🧠: {parsed_output.get('content')}")
            continue
        
        if parsed_output['step'] == 'action':
            tool_name = parsed_output.get('function')
            tool_input = parsed_output.get('input')

            if available_tools.get(tool_name, False) != False:
                output = available_tools[tool_name].get('fn')(tool_input)
                messages.append({ 'role': 'assistant', 'content': json.dumps({ 'step': 'observe', 'output': output }) })
                continue

        if parsed_output['step'] == 'output':
            print(f"🤖: {parsed_output.get('content')}")
            break