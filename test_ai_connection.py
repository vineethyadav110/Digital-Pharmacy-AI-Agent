import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load the hidden API key from the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print("Initializing AI Connection...")

try:
    # 2. Create the AI "Brain"
    # We are using Gemini 1.5 Flash, which is incredibly fast and perfect for real-time agents
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

    print("Asking the AI a medical question...")

    # 3. Send a prompt to the AI
    prompt = "You are a helpful pharmacy assistant. In one sentence, what is the active ingredient in Tylenol?"
    response = llm.invoke(prompt)

    # 4. Print the result
    print("\n✅ Success! The AI responded:")
    print("-" * 50)
    print(response.content)
    print("-" * 50)

except Exception as e:
    print(f"\n❌ Connection failed. Error: {e}")
    print("Mentor Tip: Double-check your .env file to ensure the API key is correct and has no quotes around it.")