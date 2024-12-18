# Import necessary libraries
import pdfplumber
import fitz  # PyMuPDF
import easyocr
from sentence_transformers import SentenceTransformer
import faiss
import openai
import numpy as np
import camelot
import spacy
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

# Set OpenAI API key
openai.api_key = "OPENAI_API_KEY"

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained embedding model
nlp = spacy.load("en_core_web_sm")  # NLP model for intent detection
reader = easyocr.Reader(['en'])  # EasyOCR for image-based text extraction

# Initialize FAISS index
dimension = 384  # Dimension of embeddings
index = faiss.IndexFlatL2(dimension)

# Metadata storage
metadata = []

# Function to extract text from PDF
def extract_pdf_data(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                chunk_size = 200
                chunk_list = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
                for chunk in chunk_list:
                    chunks.append((chunk, page_num))
    return chunks

# Function to extract tables using Camelot
def extract_tables(pdf_path, page_numbers):
    tables = []
    for page_num in page_numbers:
        try:
            # Specify parsing flavor for better flexibility
            table = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='stream')  # or 'lattice'
            for t in table:
                if not t.df.empty:  # Ensure the table is not empty
                    tables.append({'page_num': page_num, 'data': t.df})
        except Exception as e:
            print(f"Error extracting table from page {page_num}: {e}")
    return tables

# Function to extract images and run OCR
def extract_images_and_ocr(pdf_path):
    ocr_results = {}
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        
        # Check if the page contains images
        images = page.get_images(full=True)
        if not images:
            continue  # Skip OCR if no images are present
        
        # Convert page to image
        pix = page.get_pixmap()
        image_path = f"page_{page_num + 1}.png"
        
        # Save image
        Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(image_path)
        
        # Run OCR on the image
        results = reader.readtext(image_path)
        text = "\n".join([result[1] for result in results])
        ocr_results[page_num + 1] = text

    return ocr_results


# Index chunks in FAISS and store metadata
def index_chunks(chunks, doc_name):
    for chunk, page_num in chunks:
        embedding = embedding_model.encode(chunk)
        index.add(np.array([embedding]))
        metadata.append({'content': chunk, 'page_num': page_num, 'doc_name': doc_name})

# Process multiple PDFs
def process_multiple_pdfs(pdf_paths):
    for pdf_path in pdf_paths:
        doc_name = pdf_path.split("/")[-1]
        
        # Extract text chunks and index them
        chunks = extract_pdf_data(pdf_path)
        index_chunks(chunks, doc_name)
        
        # Extract tables and index table data
        table_data = extract_tables(pdf_path, range(1, len(chunks) + 1))
        for table in table_data:
            table_content = table['data'].to_string(index=False)  # Convert DataFrame to string
            index_chunks([(table_content, table['page_num'])], doc_name)
        
        # Extract and print OCR results from images
        ocr_results = extract_images_and_ocr(pdf_path)
        for page, text in ocr_results.items():
            print(f"OCR Text from Page {page}:\n{text}")
            index_chunks([(text, page)], doc_name)

# Detect intent of the query
def detect_intent(query):
    doc = nlp(query)
    if "compare" in query.lower():
        return "comparison"
    elif "summarize" in query.lower():
        return "summary"
    return "general"

# Query FAISS for relevant chunks
def query_pdf(query):
    intent = detect_intent(query)
    query_embedding = embedding_model.encode(query)
    D, I = index.search(np.array([query_embedding]), k=5)
    results = [metadata[i] for i in I[0] if 0 <= i < len(metadata)]
    return results, intent

# Generate prompt for GPT
def generate_prompt(query, results, intent):
    context = "\n".join([f"Page {r['page_num']} ({r['doc_name']}): {r['content']}" for r in results])
    if intent == "comparison":
        return f"Compare the following information:\n\nContext:\n{context}\n\nQuery:\n{query}"
    return f"Answer the following query based on the context:\n\nContext:\n{context}\n\nQuery:\n{query}"

# Generate response using GPT
def generate_response(query, results, intent):
    prompt = generate_prompt(query, results, intent)
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

# Streamlit UI
st.title("Chat with PDF Using RAG Pipeline")

# File upload
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")

if uploaded_files:
    pdf_paths = []
    for uploaded_file in uploaded_files:
        pdf_path = f"./{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_paths.append(pdf_path)

    # Process PDFs
    process_multiple_pdfs(pdf_paths)
    st.success("PDFs processed successfully, including tabular data!")

# Query input
query = st.text_input("Ask a question about the PDFs:")
if query:
    results, intent = query_pdf(query)
    response = generate_response(query, results, intent)
    st.write("### Response:")
    st.write(response)
