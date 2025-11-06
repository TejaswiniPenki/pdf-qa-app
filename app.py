import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
import os
import asyncio

# Fix for asyncio loop conflicts in Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set the Google API Key in the environment
st.set_page_config(page_title="PDF QA App", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #eef2f3;
        color: #2c3e50;
        font-family: Arial, sans-serif;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-size: 16px;
        padding: 8px 16px;
        border: none;
        border-radius: 6px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stFileUploader {
        margin-bottom: 20px;
    }
    .stTextArea {
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.title("📄 PDF QA App with Generative AI")
st.write("Upload a PDF document, ask a question, and get accurate, detailed responses.")

# Input: Google API Key
api_key = st.text_input(
    "Enter your Google API key:",
    type="password",
    help="Your API key is required to use the Google Generative AI services.",
)

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# File Upload Section
st.sidebar.header("📂 Upload a PDF")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file to extract content:",
    type=["pdf"],
)

# User Question Section
st.sidebar.header("📝 Ask a Question")
user_question = st.sidebar.text_area(
    "Enter your question:",
    placeholder="E.g., What is the main topic of the document?",
)

@st.cache_data
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(vector_store):
    """
    Creates a RetrievalQA chain instead of the deprecated load_qa_chain.
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

# Process Uploaded File
if uploaded_file and api_key and user_question:
    with st.spinner("Processing your PDF..."):
        try:
            raw_text = extract_text_from_pdf(uploaded_file)
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)

            # Use RetrievalQA chain instead of deprecated load_qa_chain
            chain = get_conversational_chain(vector_store)
            response = chain.run(user_question)

            # Display results
            st.success("Answer Generated!")
            st.subheader("Your Question:")
            st.write(user_question)

            st.subheader("Generated Answer:")
            st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    if not api_key:
        st.warning("Please provide your Google API key.")
    if not uploaded_file:
        st.warning("Please upload a PDF file.")
    if not user_question:
        st.warning("Please enter a question.")

# Footer
st.markdown(
    """
    ---
    🌟 Powered by LangChain and Google Generative AI
    """
)
