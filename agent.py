import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Secure API Key Loading
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print("Waking up the Digital Pharmacy Agent...")

# 2. Reconnect to our Expanded Vector Database
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
vector_db = Chroma(persist_directory="./pharmacy_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# 3. Initialize the AI Brain
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# 4. Create the Smart Prompt (With Empathy and Flow Control!)
template = """You are a highly intelligent, empathetic, and conversational digital pharmacy assistant. 

Instructions:
1. Empathy & Flow: Treat the patient with care. Do NOT repeatedly say "Hi" or "Hey there" in every response; treat this as an ongoing, continuous conversation.
2. Casual Language & Slang: Intelligently interpret everyday language, casual phrasing, or slang (e.g., "my head is pounding", "stuffed up") into actual symptoms.
3. Product Recommendations: When the patient describes a need, use ONLY the following product context to recommend a solution. 
4. Out of Stock (APOLOGIZE): If they ask for something NOT in the context, you MUST apologize empathetically. Say something natural like, "I'm so sorry you're dealing with that, but unfortunately we don't carry..." Make them feel heard, rather than just acting like a search engine.
5. Goodbyes: If the patient indicates they are done, warmly close the conversation. Do NOT ask if they need anything else.

Context: {context}
Patient's Input: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)


# Helper function to extract hidden Metadata
def format_docs(docs):
    formatted_text = []
    for doc in docs:
        name = doc.metadata.get('product_name', 'Unknown Product')
        price = doc.metadata.get('price', 'Unknown Price')
        desc = doc.page_content
        formatted_text.append(f"Product Name: {name}\nDescription: {desc}\nPrice: ${price}")
    return "\n\n".join(formatted_text)


# 5. The Modern LCEL Chain
agent_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

print("\n🤖 AI Agent Ready! Type 'quit' to exit.")
print("-" * 50)

# 6. Create the Chat Loop (Now with real-time Streaming!)
while True:
    user_input = input("\nPatient: ")

    if user_input.lower() in ['quit', 'exit']:
        print("Closing the pharmacy. Have a healthy day!")
        break

    try:
        # Print the prefix first, keeping the cursor on the same line
        print("\nPharmacy AI: ", end="", flush=True)

        # Stream the response word-by-word directly from the AI brain
        for chunk in agent_chain.stream(user_input):
            print(chunk, end="", flush=True)

        print()  # Adds a clean line break when the AI finishes typing

    except Exception as e:
        print(f"\nError: {e}")