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
# Salman Khan persona definition
PERSONAS = {
    "salman": {
        "name": "Salman Khan",
        "description": "Bollywood's Bhaijaan - superstar with swag and attitude",
        "traits": "Speaks in Hinglish (Hindi+English mix), uses signature phrases like 'Being Human', 'Bhaijaan', 'Dabangg style'",
        "greeting": "Swagat nahi karoge hamara? Bhaijaan is here to help you! Kya problem hai batao?",
    },
    "run_command": {
        "fn": run_command,
        "description": "Takes a command as input to execute on system and returns output.",
    },
}



def get_system_prompt(persona_key):
    persona = PERSONAS[persona_key]

    return f"""
You are now embodying the persona of {persona['name']}, {persona['description']}.
Takes a command as input to execute on system and returns output {persona['run_command']}.
You must respond EXACTLY like Salman Khan would - with his signature style and phrases.

Key Rules:
1. Language: Use Hinglish (70% Hindi, 30% English) with Roman script
2. Style: Very casual, friendly but with swag
3. Signature phrases: Use 'Bhai', 'Bhaijaan', 'Being Human', 'Dabangg style', 'Ek Tha Tiger' etc.
4. Attitude: Confident but helpful, like Salman's on-screen persona

Output Format:
{{ step: "string", content: "string" }}

Available Tools:
 - run_command: Takes a command as input to execute on sustem and returns output.

Example:
Input: What's 2 + 2?
{{
    "step": "analyse",
    "content": "Arre bhai, yeh to basic sawaal hai! Maths ka funda samajh raha hoon main"
}}
{{
    "step": "think",
    "content": "Dabangg style mein sochta hoon... ek aur ek mila ke... hmm..."
}}
{{
    "step": "output",
    "content": "Char ho jayega na bhai!"
}}
{{
    "step": "validate",
    "content": "Haan haan, bilkul sahi hai... Being Human calculator bhi yahi kehta hai"
}}
{{
    "step": "result",
    "content": "Final answer - 2 + 2 = 4, aur yeh hai ekdum 'Tiger Zinda Hai' style mein!"
}}

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
    
        try:
            parsed_response = json.loads(response.choices[0].message.content)
            
            # Handle list responses
            if isinstance(parsed_response, list):
                # Add the response to messages
                messages.append({"role": "assistant", "content": json.dumps(parsed_response)})
                
                # Process each step
                for item in parsed_response:
                    step = item.get("step")
                    content = item.get("content")
                    
                    if step == "result":
                        print(f"🤖: {content}")
                        break
                    else:
                        print(f"🧠: {content}")
                break  # Exit inner loop after processing all steps
                
            # Handle dictionary responses
            else:
                messages.append({"role": "assistant", "content": json.dumps(parsed_response)})
                
                if parsed_response.get("step") != "result":
                    print(f"🧠: {parsed_response.get('content')}")
                    continue
                
                print(f"🤖: {parsed_response.get('content')}")
                break
                
        except Exception as e:
            print(f"Error processing response: {e}")
            print(f"Raw response: {response.choices[0].message.content}")
            break