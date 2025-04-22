from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
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

# pdf_path = Path(__file__).parent / "nodejs.pdf"
# loader = PyPDFLoader(pdf_path)

# print("Loading PDF...")
# doc = loader.load()


# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#    chunk_size=1000, chunk_overlap=200
# )
# texts = text_splitter.split_documents(documents=doc)


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key,
)

# print("texts",embeddings)


qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60
)


# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url=os.getenv("QDRANT_URL"),
#     api_key=os.getenv("QDRANT_API_KEY"),
#     collection_name="learning_langchain",
#     embedding=embeddings,
# )

# vector_store.add_documents(documents=texts)

# print("Added documents to Qdrant collection")
# print("Collection name:", vector_store.collection_name)

retriver = QdrantVectorStore.from_existing_collection(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name="learning_langchain",
    embedding=embeddings,
)
query = input("Enter your query: ")

search_chunks = retriver.similarity_search(
    query=query,
    k=3,
)

# print("Search result:", search_chunks)


SYSTEM_PROMPT = """
You are an AI assistant whose name is ChaiCode. You are a helpful assistant that helps the user to learn programming languages. You are very good at answering questions and providing information about programming languages. You are also very good at providing code examples and explanations. You are very good at providing information about programming languages and their features. You are very good at providing information about programming languages and their features. You are very good at providing information about programming languages and their features.

Context:
{search_chunks}

"""

messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT},
]


messages.append({'role': 'user', 'content': query})

while True:
    result = client.chat.completions.create(
        model='gemini-1.5-flash-8b-001',
        response_format={"type": "json_object"},
        messages=messages
    )
    try:
        parsed_response = json.loads(result.choices[0].message.content)
        
        found = False
        for key, items in parsed_response.items():
            if isinstance(items, list) and items:
                first_description = items[0].get("description", "No description provided.")
                print(f"\n--- {key.replace('_', ' ').title()} ---\n")
                print(f"1. {first_description}")
                found = True
                break
            else:
                print(f"{key}: {items}")
        
        if found:
            break

    except json.JSONDecodeError as e:
        print("Error parsing response as JSON:", e)
        print("Raw response content:", result.choices[0].message.content)
        break