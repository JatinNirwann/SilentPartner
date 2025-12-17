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
        
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

POPPLER_PATH = os.environ.get("POPPLER_PATH")

if not POPPLER_PATH and os.path.exists(LOCAL_POPPLER_PATH):
    POPPLER_PATH = LOCAL_POPPLER_PATH

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEBUG: Using device: {DEVICE}")

from PIL import Image, ImageOps, ImageEnhance
import numpy as np

def preprocess_image(image):

    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image = image.convert('L')
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    return image

def clean_ocr_text(text):

    if not text:
        return ""
    
    text = text.replace('|', '')
    
    text = re.sub(r'(\d+)\s+,\s+(\d{2})', r'\1.\2', text)
    
    boilerplate = [
        r'page \d+ of \d+',
        r'scanned by',
        r'thank you.*come again',
        r'continued on next page'
    ]
    for p in boilerplate:
        text = re.sub(p, '', text, flags=re.IGNORECASE)

    text = re.sub(r'\n{3,}', '\n\n', text)
    
    lines = text.split('\n')
    seen = set()
    deduped_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and stripped in seen:
            continue
        if stripped:
            seen.add(stripped)
        deduped_lines.append(stripped)
    text = '\n'.join(deduped_lines)
    
    return text.strip()

    return text.strip()

def run_ocr_on_image(image):

    processed_image = preprocess_image(image)
    
    text = pytesseract.image_to_string(processed_image)
    text = clean_ocr_text(text)
    
    is_good_quality = len(text) > 50 
    
    if is_good_quality:
        print("DEBUG: Tesseract result acceptable.")
        return text
    
    if HAS_EASYOCR:
        print("DEBUG: Tesseract result poor, falling back to EasyOCR...")
        try:
            img_np = np.array(processed_image)
            easy_results = reader.readtext(img_np, detail=0)
            easy_text = " ".join(easy_results)
            
            if len(easy_text) > len(text):
                return easy_text
        except Exception as e:
            print(f"EasyOCR fallback failed: {e}")
            
    return text

def extract_text_from_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH) if POPPLER_PATH else convert_from_path(pdf_path, dpi=300)
        text = ""
        for i, image in enumerate(images):
            page_text = run_ocr_on_image(image)
            text += f"--- Page {i+1} ---\n{page_text}\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return f"Error extracting text from PDF: {str(e)}. Ensure Poppler is installed and configured."

def extract_text_from_image(image_path):
    try:
        return run_ocr_on_image(image_path)
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
    
    amount_pattern = r'[\$€£₹]\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
    amounts = re.findall(amount_pattern, text)
    if amounts:
        try:
            float_amounts = [float(a.replace(',', '')) for a in amounts]
            metadata['potential_total'] = max(float_amounts) if float_amounts else 0
            metadata['amounts'] = amounts
        except:
            pass

    inv_pattern = r'(?:Invoice|Inv|Bill)(?:[ \t]+(?:No|Number|#))?[ \t.:\-#]+([A-Za-z0-9\-\/_]{3,})'
    inv_match = re.search(inv_pattern, text, re.IGNORECASE)
    if inv_match:
        metadata['invoice_number'] = inv_match.group(1)

    tax_pattern = r'(?:Tax|GST|VAT)\s*[:\-]?\s*[\$€£₹]?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
    tax_match = re.search(tax_pattern, text, re.IGNORECASE)
    if tax_match:
        metadata['tax_amount'] = tax_match.group(1)

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
        
    # Extract metadata
    metadata = extract_metadata(text)
    
    print(f"DEBUG: Extracted text length: {len(text)}")
    print(f"DEBUG: Text preview: {text[:100]}")
    
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

load_index()

def smart_chunk(text, chunk_size=1000, overlap=100):
    
    if not text:
        return []
    
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        is_table_like = len(re.findall(r'\n', para)) > 3 and (len(para) / len(re.findall(r'\n', para)) < 50)
        
        if len(current_chunk) + len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                # Use the para as is (too big?)
                chunks.append(para.strip())
                current_chunk = "" 
        else:
            current_chunk += "\n\n" + para
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def generate_embeddings(text):
    if not text or text.startswith("Error"):
        return [], []
    
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    if len(clean_text) < 10:
         print("DEBUG: Skipping embedding generation for noisy/empty text.")
         return [], []

    chunks = smart_chunk(text, chunk_size=1000, overlap=150) 
    
    if not chunks:
        return [], []
        
    embeddings = embedding_model.encode(chunks)
    return chunks, embeddings

def add_to_index(chunks, embeddings):
    global index
    if len(embeddings) == 0:
        return index.ntotal
        
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    save_index()
    return index.ntotal

def search_index(query_text, k=5):
    if index.ntotal == 0:
        return [], []
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
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        system_instruction = """You are a helpful document assistant. Your task is to answer the user's question based ONLY on the provided context chunks.

Rules:
1. Answer ONLY from the retrieved context.
2. If the answer is not in the context, say EXACTLY: "I cannot find this in your uploaded documents."
3. Ignore OCR error text and noise.
4. Do not limit yourself to the first chunk; synthesize information from all provided chunks.
5. Provide a direct answer without robotic preambles.
6. If the context contains a table or list, try to preserve the structure in your answer.
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

Rules:
1. Answer ONLY from retrieved document chunks.
2. If answer not found, say: "I cannot find this in your uploaded documents."
3. Ignore OCR error text.
4. No assumptions, no external knowledge.

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
