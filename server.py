from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import threading
from werkzeug.utils import secure_filename
from rag_engine import process_document, rag_query
from database import init_db, save_document, save_chunks, get_db, save_chat, get_chats, delete_chat_history, get_document_by_id

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Lock for DB writes
db_lock = threading.Lock()

# Initialize DB on startup
with app.app_context():
    init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        os.makedirs("data/documents", exist_ok=True)
        filename = secure_filename(file.filename)
        file_path = os.path.join("data/documents", filename)
        file.save(file_path)
        
        # Process Document
        result = process_document(file_path)
        extracted_text = result["text"]
        metadata = result["metadata"]
        file_size = os.path.getsize(file_path)
        
        with db_lock:
            doc_id = save_document(filename, file_path, file_size, extracted_text, metadata)
            
            # RAG Processing
            from rag_engine import generate_embeddings, add_to_index, index
            chunks, embeddings = generate_embeddings(extracted_text)
            
            if chunks:
                start_faiss_id = index.ntotal
                add_to_index(chunks, embeddings)
                save_chunks(doc_id, chunks, start_faiss_id)
                
        return jsonify({
            "filename": filename,
            "status": "processed",
            "doc_id": doc_id,
            "text_preview": extracted_text[:100] + "..." if extracted_text else "No text extracted",
            "metadata": metadata
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400
        
    result = rag_query(query)
    
    with db_lock:
        save_chat(query, result["response"])
        
    return jsonify(result)

@app.route("/chats", methods=["GET"])
def get_history():
    return jsonify({"chats": get_chats()})

@app.route("/chats", methods=["DELETE"])
def clear_history():
    with db_lock:
        delete_chat_history()
    return jsonify({"status": "cleared"})
    
@app.route("/documents", methods=["GET"])
def list_docs():
    db = get_db()
    try:
        results = list(db["documents"].rows)
        return jsonify({"documents": results})
    except Exception:
        return jsonify({"documents": []})

@app.route("/documents/<int:doc_id>", methods=["DELETE"])
def delete_doc(doc_id):
    with db_lock:
        db = get_db()
        db["documents"].delete(doc_id)
        # Also delete chunks - safe way if table doesn't exist
        try:
            db["chunks"].delete_where("document_id = ?", [doc_id])
        except Exception:
            pass
    return jsonify({"status": "deleted"})

@app.route("/documents/<int:doc_id>/file", methods=["GET"])
def get_document_file(doc_id):
    """Serve the original uploaded document file."""
    doc = get_document_by_id(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    
    file_path = doc.get("path")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found on disk"}), 404
    
    return send_file(file_path, as_attachment=False)

@app.route("/documents/<int:doc_id>/text", methods=["GET"])
def get_document_text(doc_id):
    """Return the extracted text for a document."""
    doc = get_document_by_id(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    
    return jsonify({
        "id": doc_id,
        "filename": doc.get("filename"),
        "extracted_text": doc.get("extracted_text", ""),
        "metadata": doc.get("metadata")
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
