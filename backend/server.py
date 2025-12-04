from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Status": "Backend is running"}

from rag_engine import process_document
from database import init_db, save_document, save_chunks, get_db
from datetime import datetime

@app.on_event("startup")
def startup_event():
    init_db()

import asyncio

db_lock = asyncio.Lock()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        os.makedirs("data/documents", exist_ok=True)
        
        file_path = os.path.join("data/documents", file.filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        result = process_document(file_path)
        extracted_text = result["text"]
        metadata = result["metadata"]
        
        file_size = os.path.getsize(file_path)
        
        async with db_lock:
            doc_id = save_document(file.filename, file_path, file_size, extracted_text, metadata)
            
            from rag_engine import generate_embeddings, add_to_index, index
            chunks, embeddings = generate_embeddings(extracted_text)
            
            if chunks:
                start_faiss_id = index.ntotal
                add_to_index(chunks, embeddings)
                save_chunks(doc_id, chunks, start_faiss_id)
        
        return {
            "filename": file.filename, 
            "status": "processed", 
            "path": file_path,
            "doc_id": doc_id,
            "metadata": metadata,
            "chunks_count": len(chunks),
            "text_preview": extracted_text[:100] + "..." if extracted_text else "No text extracted"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

from rag_engine import rag_query
from database import save_chat, get_chats, delete_chat_history

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    result = await asyncio.to_thread(rag_query, request.query)
    
    async with db_lock:
        save_chat(request.query, result["response"])
        
    return result

@app.get("/chats")
def get_chat_history_endpoint():
    return {"chats": get_chats()}

@app.delete("/chats")
async def clear_chat_history():
    async with db_lock:
        delete_chat_history()
    return {"status": "cleared"}

@app.get("/search")
def search_documents(q: str):
    db = get_db()
    results = list(db["documents"].search(q))
    return {"query": q, "results": results}

@app.get("/documents")
def list_documents():
    db = get_db()
    results = list(db["documents"].rows)
    return {"documents": results}

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    async with db_lock:
        db = get_db()
        db["documents"].delete(doc_id)
        db["chunks"].delete_where("document_id = ?", [doc_id])
    return {"status": "deleted"}

@app.get("/reminders")
def get_reminders():
    db = get_db()
    results = list(db.query("SELECT * FROM documents WHERE expiry_date IS NOT NULL AND expiry_date > date('now') ORDER BY expiry_date ASC"))
    return {"reminders": results}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
