import streamlit as st
import json
import os
import sys
import subprocess

# Ensure necessary modules are installed
try:
    from langchain.schema import HumanMessage, SystemMessage
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain"])
    from langchain.schema import HumanMessage, SystemMessage  # Retry import

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain_huggingface"])
    from langchain_huggingface import HuggingFaceEmbeddings  # Retry import

try:
    from langchain_groq import ChatGroq
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain_groq"])
    from langchain_groq import ChatGroq  # Retry import

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer  # Retry import

try:
    import chromadb
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chromadb"])
    import chromadb  # Retry import

# JSON Memory File
MEMORY_FILE = "chat_memory.json"

def load_chat_history():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as file:
            return json.load(file)
    return []

def save_chat_history(chat_history):
    with open(MEMORY_FILE, "w") as file:
        json.dump(chat_history, file)

# Initialize Memory
memory = load_chat_history()

# Initialize Models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="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")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

def query_llama3(user_query):
    system_prompt = "System Prompt: Your AI clone personality based on Manas Patni."
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]
    try:
        response = chat.invoke(messages)
        memory.append({"input": user_query, "output": response.content})
        save_chat_history(memory)
        return response.content
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

def main():
    st.title("AI Chatbot Based on Manas Patni")
    st.markdown("Welcome to the AI chatbot interface. Ask a question to get started!")
    st.markdown("### Chat History")
    if memory:
        for chat in memory:
            st.markdown(
                f"""
                <div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>
                    <div style='background-color: #d1e7dd; padding: 10px; border-radius: 10px; max-width: 60%;'>
                        <strong>You:</strong> {chat['input']}
                    </div>
                </div>
                <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                    <div style='background-color: #f8d7da; padding: 10px; border-radius: 10px; max-width: 60%;'>
                        <strong>AI:</strong> {chat['output']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.write("No previous chat history.")

    user_query = st.text_input("Enter your question:")
    if user_query:
        response = query_llama3(user_query)
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>
                <div style='background-color: #d1e7dd; padding: 10px; border-radius: 10px; max-width: 60%;'>
                    <strong>You:</strong> {user_query}
                </div>
            </div>
            <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                <div style='background-color: #f8d7da; padding: 10px; border-radius: 10px; max-width: 60%;'>
                    <strong>AI:</strong> {response}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
