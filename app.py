import os
import streamlit as st
import tempfile
from PIL import Image
import easyocr
import numpy as np
import fitz  # PyMuPDF
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import pypdf
import google.generativeai as genai
from dotenv import load_dotenv

# === Environment and OpenMP Settings ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

load_dotenv()

# === Streamlit Page Config ===
st.set_page_config(page_title="PDF Chat App", layout="wide")
st.title("Chat with your PDF Documents")
st.markdown("Upload a PDF, ask questions, and get answers based on the document content")

# === Global State Initialization ===
if 'reader' not in st.session_state:
    st.session_state.reader = None

if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session" not in st.session_state:
    st.session_state.current_session = "Session 1"
    st.session_state.sessions["Session 1"] = {
         "document_text": "",
         "db": None,
         "processed": False,
         "chat_history": [],
         "ocr_applied": False,
         "previous_uploaded_file": None
    }

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

def get_current_session():
    return st.session_state.sessions[st.session_state.current_session]

# === PDF Extraction with Caching ===
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_bytes):
    pdf_reader = pypdf.PdfReader(pdf_bytes)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# === OCR Function ===
def perform_ocr(pdf_file_path):
    if st.session_state.reader is None:
        with st.spinner("Initializing OCR engine..."):
            st.session_state.reader = easyocr.Reader(['en'], gpu=False)
    text = ""
    doc = fitz.open(pdf_file_path)
    progress_bar = st.progress(0)
    total_pages = len(doc)
    for page_num in range(total_pages):
        progress_bar.progress((page_num + 1) / total_pages)
        try:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)
            results = st.session_state.reader.readtext(img_np, detail=0)
            text += " ".join(results) + "\n\n"
        except Exception as e:
            st.warning(f"Error processing page {page_num+1}: {e}")
    progress_bar.empty()
    return text

# === Document Processing ===
def process_document(file_path, session_data):
    with st.spinner("Processing document..."):
        try:
            if not session_data["ocr_applied"]:
                # Use caching for PDF extraction
                text = extract_text_from_pdf(file_path)
                if len(text.strip()) < 100:
                    st.warning("Applying OCR...")
                    text = perform_ocr(file_path) or text
                    session_data["ocr_applied"] = True
                    st.success("OCR applied successfully!")
                if not text.strip():
                    st.error("No text could be extracted.")
                    return False
                session_data["document_text"] = text
            else:
                text = session_data["document_text"]

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)
            
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                st.error("Google API key is missing.")
                return False
            
            genai.configure(api_key=api_key)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            session_data["db"] = FAISS.from_texts(chunks, embeddings) if chunks else None
            session_data["processed"] = True
            st.success("Document processed successfully!")
            return True
        except Exception as e:
            st.error(f"Error processing document: {e}")
            return False

# === PDF Removal Helpers ===
def clear_pdf(session_data):
    session_data["processed"] = False
    session_data["document_text"] = ""
    session_data["db"] = None
    session_data["chat_history"] = []
    session_data["ocr_applied"] = False
    session_data["previous_uploaded_file"] = None
    st.info("PDF removed successfully.")

def confirm_pdf_removal(session_data):
    if session_data["previous_uploaded_file"]:
        if st.button("Remove PDF"):
            clear_pdf(session_data)

# === Sidebar: Sessions and Document Upload ===
with st.sidebar:
    st.header("Chat Sessions")
    if st.button("New Chat"):
        new_session_name = f"Session {len(st.session_state.sessions) + 1}"
        st.session_state.sessions[new_session_name] = {
            "document_text": "",
            "db": None,
            "processed": False,
            "chat_history": [],
            "ocr_applied": False,
            "previous_uploaded_file": None
        }
        st.session_state.current_session = new_session_name
        st.session_state.question_input = ""

    session_names = list(st.session_state.sessions.keys())
    selected = st.selectbox("Select Chat Session", session_names, index=session_names.index(st.session_state.current_session))
    if selected != st.session_state.current_session:
        st.session_state.current_session = selected
        st.session_state.question_input = ""

    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    current_session = get_current_session()
    if uploaded_file:
        # Only process a new file if it differs from the previous one
        if uploaded_file != current_session["previous_uploaded_file"]:
            current_session["previous_uploaded_file"] = uploaded_file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            process_document(tmp_path, current_session)
            os.unlink(tmp_path)
    else:
        confirm_pdf_removal(current_session)

# === Main Chat Interface ===
current_session = get_current_session()
if current_session["processed"]:
    st.header("Ask Questions About Your Document")
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Your question:", key="question_input", value=st.session_state.question_input)
        submitted = st.form_submit_button("Submit")
    if submitted and question:
        existing_answer = next((chat["answer"] for chat in current_session["chat_history"] if chat["question"] == question), None)
        if not existing_answer:
            try:
                docs = current_session["db"].similarity_search(question, k=3)
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3, max_output_tokens=2048)
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=question)
                current_session["chat_history"].append({"question": question, "answer": response})
                existing_answer = response
            except Exception as e:
                st.error(f"Error generating response: {e}")
                existing_answer = "Sorry, error generating response."
        st.markdown(f"**Answer:** {existing_answer}")

    if current_session["chat_history"]:
        st.markdown("### Chat History")
        for i, chat in enumerate(current_session["chat_history"]):
            with st.expander(f"Q{i+1}: {chat['question'][:50]}..."):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
        if st.button("Clear Chat History"):
            current_session["chat_history"] = []
else:
    st.info("Upload a document to start.")
