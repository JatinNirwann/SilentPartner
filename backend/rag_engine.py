import pytesseract
from pdf2image import convert_from_path
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_POPPLER_PATH = os.path.join(BASE_DIR, "deps", "poppler", "Library", "bin")

TESSERACT_PATH = os.environ.get("TESSERACT_PATH")

if not TESSERACT_PATH and os.name == 'nt':
    common_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(common_path):
        TESSERACT_PATH = common_path

POPPLER_PATH = os.environ.get("POPPLER_PATH")

if not POPPLER_PATH and os.path.exists(LOCAL_POPPLER_PATH):
    POPPLER_PATH = LOCAL_POPPLER_PATH

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEBUG: Using device: {DEVICE}")

try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=(DEVICE == "cuda")) 
    HAS_EASYOCR = True
    print(f"DEBUG: EasyOCR initialized successfully (GPU: {DEVICE == 'cuda'})")
except ImportError:
    HAS_EASYOCR = False
    print("DEBUG: EasyOCR not found. Install 'easyocr' for handwriting support.")
except Exception as e:
    HAS_EASYOCR = False
    print(f"DEBUG: EasyOCR initialization failed: {e}")

def extract_text_from_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH) if POPPLER_PATH else convert_from_path(pdf_path)
        text = ""
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            
            if HAS_EASYOCR:
                try:
                    import numpy as np
                    img_np = np.array(image)
                    easy_results = reader.readtext(img_np, detail=0)
                    easy_text = " ".join(easy_results)
                    page_text += f"\n\n[Handwriting/Complex Text]:\n{easy_text}"
                except Exception as e:
                    print(f"EasyOCR error on page {i}: {e}")

            text += f"--- Page {i+1} ---\n{page_text}\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return f"Error extracting text from PDF: {str(e)}. Ensure Poppler is installed and configured."

def extract_text_from_image(image_path):
    try:
        text = pytesseract.image_to_string(image_path)
        
        if HAS_EASYOCR:
            try:
                easy_results = reader.readtext(image_path, detail=0)
                easy_text = " ".join(easy_results)
                text += f"\n\n[Handwriting/Complex Text]:\n{easy_text}"
            except Exception as e:
                print(f"EasyOCR error on image: {e}")
                
        return text
    except pytesseract.TesseractNotFoundError:
        return "Error: Tesseract OCR is not installed or not in your PATH. Please install Tesseract."
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return f"Error extracting text from image: {str(e)}"

import re
from dateutil.parser import parse

def extract_metadata(text):
    metadata = {}
    if not text or text.startswith("Error"):
        return metadata
        
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{2}/\d{2}/\d{4}',
        r'\d{2}-\d{2}-\d{4}',
        r'\d{1,2} [A-Za-z]+ \d{4}'
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                dt = parse(match, fuzzy=True)
                dates.append(dt.isoformat())
            except:
                pass
    
    if dates:
        metadata['dates'] = list(set(dates))
        try:
            from datetime import datetime
            now = datetime.now()
            future_dates = [d for d in dates if parse(d) > now]
            if future_dates:
                metadata['expiry_date'] = max(future_dates)
        except:
            pass
        
    amount_pattern = r'[\$€£₹]\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
    amounts = re.findall(amount_pattern, text)
    if amounts:
        metadata['amounts'] = amounts

    return metadata

def process_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        text = extract_text_from_image(file_path)
    else:
        text = "Unsupported file format"
        
    metadata = extract_metadata(text)
    
    return {
        "text": text,
        "metadata": metadata
    }

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
dimension = 384
index = faiss.IndexFlatL2(dimension)

def save_index(path="data/faiss_index.bin"):
    faiss.write_index(index, path)

def load_index(path="data/faiss_index.bin"):
    global index
    if os.path.exists(path):
        index = faiss.read_index(path)
    else:
        index = faiss.IndexFlatL2(dimension)

# Initialize index on load
load_index()

def generate_embeddings(text):
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    if not chunks:
        return [], []
        
    embeddings = embedding_model.encode(chunks)
    return chunks, embeddings

def add_to_index(chunks, embeddings):
    global index
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    save_index()
    return index.ntotal

def search_index(query_text, k=5):
    query_embedding = embedding_model.encode([query_text])
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, k)
    return I[0], D[0]

import requests

import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def query_gemini(prompt, context):
    try:
        if not GEMINI_API_KEY:
            return None
        
        model = genai.GenerativeModel('gemini-pro')
        system_instruction = """You are a helpful document assistant. Your task is to answer the user's question based ONLY on the provided context. 
        
Rules:
1. If the context contains error messages (like "Error: Tesseract OCR is not installed"), IGNORE them and state that the document content could not be read due to an error.
2. If the answer is not in the context, say "I cannot find the answer in your documents."
3. Do not make up information.
4. Be concise and direct.
"""
        full_prompt = f"{system_instruction}\n\nContext:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return None

def query_llm(prompt, context):
    gemini_response = query_gemini(prompt, context)
    if gemini_response:
        return gemini_response

    try:
        url = "http://localhost:11434/api/generate"
        data = {
            "model": "llama3",
            "prompt": f"""You are a helpful document assistant. Answer based ONLY on the context.
Context:
{context}

Question: {prompt}

Answer:""",
            "stream": False
        }
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json().get("response", "")
    except:
        pass
    
    return "Error: Could not connect to local LLM (Ollama) and no Gemini API key found.\n\nTo use Ollama: Run 'ollama serve'.\nTo use Gemini: Set GEMINI_API_KEY environment variable."

from database import get_chunk_by_index

def rag_query(query_text):
    indices, distances = search_index(query_text)
    
    context_parts = []
    for idx in indices:
        if idx == -1: continue
        chunk = get_chunk_by_index(idx)
        if chunk:
            context_parts.append(chunk["content"])
            
    context = "\n---\n".join(context_parts)
    
    answer = query_llm(query_text, context)
    
    sources = []
    for idx in indices:
        if idx == -1: continue
        chunk = get_chunk_by_index(idx)
        if chunk:
            sources.append({
                "source": chunk["filename"],
                "text": chunk["content"][:200] + "...",
                "page": 1
            })

    return {
        "response": answer,
        "sources": sources
    }
