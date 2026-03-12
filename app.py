import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration
st.set_page_config(page_title="AI Pharmacy Assistant", page_icon="💊", layout="centered")
st.title("💊 Digital Pharmacy Assistant")
st.markdown("Welcome! Describe your symptoms or ask for a product, and I'll find the right remedy for you.")

# 2. Secure API Key Loading
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


# 3. Cache the AI Setup (So it doesn't reload on every single message)
@st.cache_resource
def load_ai_agent():
    # Connect to Memory
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    vector_db = Chroma(persist_directory="./pharmacy_db", embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    # Initialize Brain
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

    # Prompt Engineering
    template = """You are a highly intelligent, empathetic, and conversational digital pharmacy assistant. 

    Instructions:
    1. Empathy & Flow: Treat the patient with care. Do NOT repeatedly say "Hi" or "Hey there".
    2. Casual Language: Interpret everyday language or slang (e.g., "my head is pounding") into actual symptoms.
    3. Product Recommendations: Use ONLY the following product context to recommend a solution. 
    4. Out of Stock: If they ask for something NOT in the context, you MUST apologize empathetically.
    5. Goodbyes: If the patient indicates they are done, warmly close the conversation.

    Context: {context}
    Patient's Input: {question}

    Helpful Answer:"""
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        formatted_text = []
        for doc in docs:
            name = doc.metadata.get('product_name', 'Unknown Product')
            price = doc.metadata.get('price', 'Unknown Price')
            desc = doc.page_content
            formatted_text.append(f"Product Name: {name}\nDescription: {desc}\nPrice: ${price}")
        return "\n\n".join(formatted_text)

    # Build Pipeline
    agent_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return agent_chain


# Load the agent
agent_chain = load_ai_agent()

# 4. Initialize Chat History in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. The Chat UI and Logic
# This creates the text box at the bottom of the screen
if prompt := st.chat_input("How can I help you feel better today?"):
    # Show user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to memory
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show AI response with streaming effect
    with st.chat_message("assistant"):
        # Streamlit's built-in function to handle LangChain streams beautifully!
        response = st.write_stream(agent_chain.stream(prompt))

    # Add AI response to memory
    st.session_state.messages.append({"role": "assistant", "content": response})