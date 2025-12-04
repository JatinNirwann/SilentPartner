import sqlite_utils
import os
from datetime import datetime

DB_PATH = "data/rag_app.db"

def get_db():
    os.makedirs("data", exist_ok=True)
    db = sqlite_utils.Database(DB_PATH)
    db.conn.execute("PRAGMA busy_timeout = 20000;")
    try:
        db.enable_wal()
    except:
        pass
    return db

def init_db():
    db = get_db()
    
    if "documents" not in db.table_names():
        db["documents"].create({
            "id": int,
            "filename": str,
            "path": str,
            "size": int,
            "upload_date": str,
            "extracted_text": str,
            "metadata": str,
            "expiry_date": str,
        }, pk="id")
        db["documents"].enable_fts(["extracted_text", "filename"], create_triggers=True)

    if "chunks" not in db.table_names():
        db["chunks"].create({
            "id": int,
            "document_id": int,
            "content": str,
            "embedding_id": int,
        }, pk="id", foreign_keys=[("document_id", "documents", "id")])

    db["chats"].create({
        "id": int,
        "query": str,
        "response": str,
        "created_at": str
    }, pk="id", if_not_exists=True)

def save_document(filename, path, size, content, metadata):
    db = get_db()
    return db["documents"].insert({
        "filename": filename,
        "path": path,
        "size": size,
        "content": content,
        "metadata": metadata,
        "created_at": datetime.now().isoformat(),
        "expiry_date": metadata.get("expiry_date")
    }).last_pk

def save_chat(query, response):
    db = get_db()
    return db["chats"].insert({
        "query": query,
        "response": response,
        "created_at": datetime.now().isoformat()
    }).last_pk

def get_chats():
    db = get_db()
    return list(db.query("SELECT * FROM chats ORDER BY created_at ASC"))

def delete_chat_history():
    db = get_db()
    db["chats"].delete_where()

def save_chunks(document_id, chunks, start_faiss_id):
    db = get_db()
    chunk_records = []
    for i, content in enumerate(chunks):
        chunk_records.append({
            "document_id": document_id,
            "content": content,
            "embedding_id": start_faiss_id + i
        })
    db["chunks"].insert_all(chunk_records)

def get_chunk_by_index(faiss_id):
    db = get_db()
    query = """
    SELECT c.content, c.document_id, d.filename 
    FROM chunks c 
    JOIN documents d ON c.document_id = d.id 
    WHERE c.embedding_id = ?
    """
    results = list(db.query(query, [int(faiss_id)]))
    if results:
        return results[0]
    return None
