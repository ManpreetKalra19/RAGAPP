import streamlit as st
import os
import tempfile
import time
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.llms import HuggingFaceHub
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(page_title="Document Q&A Bot", layout="wide")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

# Title
st.title("Document Q&A Bot")
st.write("Upload a document (PDF or DOCX) and ask questions about its content.")

# Function to process documents and create a retrieval chain
def process_document(file_path, use_openai=True, api_key=None):
    try:
        # Load the document based on file type
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file type")
        
        documents = loader.load()
        
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store based on user selection
        if use_openai and api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            embeddings = OpenAIEmbeddings()
            llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
        else:
            # Use HuggingFace embeddings as a fallback
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # If HuggingFace Hub API token is provided
            if st.session_state.get("hf_api_key"):
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state.hf_api_key
                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-base",
                    model_kwargs={"temperature": 0.1, "max_length": 512}
                )
            else:
                # Use OpenAI with warning
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
                else:
                    st.error("No language model available. Please provide either an OpenAI or HuggingFace API key.")
                    return None
        
        # Try to create the vector store
        try:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            st.error(f"Error creating vector store: {str(e)}")
            return None
        
        # Create a retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Create a conversation memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Create a conversation chain
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            get_chat_history=lambda h: h,
        )
        
        return conversation
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        st.error(f"Error processing document: {str(e)}")
        return None

# Sidebar for API key and document upload
with st.sidebar:
    st.header("Configuration")
    
    # API Keys section
    st.subheader("API Keys")
    tab1, tab2 = st.tabs(["OpenAI", "HuggingFace"])
    
    with tab1:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        use_openai = st.checkbox("Use OpenAI for embeddings", value=True)
    
    with tab2:
        hf_api_key = st.text_input("HuggingFace API Key (Optional)", type="password")
        if hf_api_key:
            st.session_state.hf_api_key = hf_api_key
    
    # Document upload
    st.subheader("Document Upload")
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx"])
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            # Create a temporary file
            temp_dir = tempfile.TemporaryDirectory()
            temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
            
            # Write the uploaded file to the temporary file
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the document
            conversation = process_document(
                temp_file_path, 
                use_openai=use_openai, 
                api_key=openai_api_key
            )
            
            if conversation:
                st.session_state.conversation = conversation
                st.session_state.document_loaded = True
                st.success(f"Document '{uploaded_file.name}' loaded successfully!")
            else:
                st.error("Failed to process document. Check the error message above.")

# Main chat interface
if st.session_state.document_loaded and st.session_state.conversation:
    st.header("Ask questions about your document")
    query = st.text_input("Ask a question:")
    
    if query:
        with st.spinner("Thinking..."):
            try:
                # Get the response from the conversation chain
                response = st.session_state.conversation.invoke({"question": query})
                
                # Store the conversation in the session state
                st.session_state.chat_history.append((query, response["answer"]))
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")
    
    # Display the chat history
    for i, (query, response) in enumerate(st.session_state.chat_history):
        st.subheader(f"Question {i+1}: {query}")
        st.write(response)
        st.divider()

else:
    if not (openai_api_key or st.session_state.get("hf_api_key")):
        st.warning("Please enter at least one API key in the sidebar (OpenAI or HuggingFace).")
    if not uploaded_file:
        st.info("Please upload a document in the sidebar to get started.")