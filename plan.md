
# RAG App – Fixes & Guidelines (Quick Reference)

## ❌ What’s wrong → ✅ What to do

### 1. Database schema mismatch
**Problem**
- Table uses `extracted_text`, `upload_date`
- Insert uses `content`, `created_at`

**Fix**
- Use consistent fields everywhere:
  - `extracted_text`
  - `created_at`

---

### 2. Poor chunking strategy
**Problem**
- Fixed-size character chunking
- Breaks sentences, tables, invoice totals

**Fix**
- Chunk by paragraphs / newlines
- Add overlap (30–50 tokens)

---

### 3. OCR fails on images inside PDFs
**Problem**
- PDF pages rendered at default ~72 DPI
- OCR receives blurry text

**Fix**
```python
convert_from_path(
    pdf_path,
    dpi=300,
    poppler_path=POPPLER_PATH
)
image = image.convert("L")
```

---

### 4. OCR works on images but not PDFs
**Cause**
- Standalone images are high resolution
- PDF-rendered images are low resolution

**Fix**
- Always render PDF pages at **300 DPI**
- Treat rendered pages exactly like normal images

---

### 5. OCR error text polluting embeddings
**Problem**
- Embedding OCR error messages

**Fix**
- If OCR fails → do NOT embed
- Mark document as unreadable

---

## 🧠 LLM Response Guidelines

### Hard rules
- Answer ONLY from retrieved document chunks
- If answer not found:
  > "I cannot find this in your uploaded documents."
- Ignore OCR error text
- No assumptions, no external knowledge

### Context control
- Limit max context size
- Deduplicate similar chunks
- Prefer fewer, higher-quality chunks

### Response style
- Short and direct
- No explanations unless asked
- No hallucinations
- No guessing

---

## ✅ Recommended Pipeline

PDF  
→ Render pages at **300 DPI**  
→ OCR  
→ Clean text  
→ Smart chunking  
→ Embeddings  
→ Retrieval  
→ **LLM constrained to context**
