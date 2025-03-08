import streamlit as st
import os
import shutil
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv


torch.classes.__path__ = []    # To avoid a error

# Ensure the directory exists
PDF_DIR = "pdf_docs"
os.makedirs(PDF_DIR, exist_ok=True)

# **Delete all documents when a new session starts**
if "session_initialized" not in st.session_state:
    shutil.rmtree(PDF_DIR)  # Delete all files
    os.makedirs(PDF_DIR, exist_ok=True)  # Recreate directory
    st.session_state.session_initialized = True  # Mark session as initialized

def create_vector_embedding():
    if "vectors" not in st.session_state:
        if not os.listdir(PDF_DIR):  # Check if folder is empty
            return  # Don't create an empty vector store
    
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("pdf_docs")  # Load PDFs
        st.session_state.docs = st.session_state.loader.load()  # Extract text from PDFs
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Split text into chunks

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Load env variables
load_dotenv()

groq_api_key = st.secrets["GROQ_API_KEY"]       # os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = st.secrets["HF_TOKEN"] # os.getenv("HF_TOKEN")

# Create LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI assistant.
    Use the provided context to answer the question. 
    If the context is not relevant or unavailable, provide a detailed answer based on your general knowledge.
    
    Context:
    {context}
    
    Question: {input}
    Answer:
    """
)

######################### Streamlit UI ##########################
st.title("Document Question Answering (Q & A) using RAG")

# PDF Upload Section
uploaded_files = st.file_uploader("Upload documents", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    uploaded_file_names = {uploaded_file.name for uploaded_file in uploaded_files}
else:
    uploaded_file_names = set()

# Delete files that were not uploaded in the current session
existing_files = set(os.listdir(PDF_DIR))

# If the uploaded files change, reset user input and context
if uploaded_file_names != existing_files:
    st.session_state["user_prompt"] = ""
    st.session_state["context"] = []

for file in existing_files:
    if file not in uploaded_file_names:
        os.remove(os.path.join(PDF_DIR, file))  # Delete file

# Save uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(PDF_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

# Create vector store
if st.button("Upload", disabled=not uploaded_files):
    create_vector_embedding()
    st.success("Document(s) uploaded successfully!")

# Accept user prompt
user_prompt = st.text_input("Enter your question", key="user_prompt")

if user_prompt:

    # Clear previous context
    if 'context' in st.session_state:
        del st.session_state["context"]

    if "vectors" in st.session_state:
        # Create retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Invoke retrieval-based response
        response = retrieval_chain.invoke({'input': user_prompt})
        # Extract the answer if available
        answer = response.get('answer', "").strip()

        if answer:  # If the context contains answer
            st.write(answer)
            st.session_state["context"] = response.get('context', [])
            # Display retrieved context if available
            if st.session_state["context"]:
                with st.expander("Context of answer"):
                    for i, doc in enumerate(st.session_state["context"]):
                        st.write(doc.page_content)
                        st.write('-' * 40)
        else:       # If the context dont contain the answer
            response = llm.invoke(f'Provide detailed answer for {user_prompt}')
            answer = response.content.strip()
    else:
        # If user didnt uploaded any file, directly query LLM
        response = llm.invoke(user_prompt)
        st.write(response.content.strip())
        
        if 'context' in st.session_state:
            del st.session_state["context"]

# Footer
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            left: 50%;
            bottom: 0px;
            transform: translateX(-50%);
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: #333;
            width: 100%;
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
        }
    </style>
    <div class="footer">
        Made with ❤️ by <b>Mayuresh Dindorkar</b>
    </div>
    """,
    unsafe_allow_html=True
)
