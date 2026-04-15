from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import time
from pypdf import PdfReader

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = []
embeddings = []

index = faiss.IndexFlatL2(384)

# ----------- Helper Functions -----------

def read_file(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def process_document(file_path):
    global documents, embeddings, index

    text = read_file(file_path)
    chunks = chunk_text(text)

    for chunk in chunks:
        emb = model.encode([chunk])[0]
        documents.append(chunk)
        embeddings.append(emb)

    index.add(np.array(embeddings))

# ----------- API -----------

@app.post("/upload/")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    background_tasks.add_task(process_document, file_path)

    return {"message": "File uploaded and processing started"}

class Query(BaseModel):
    question: str

@app.post("/query/")
def query(q: Query):
    start = time.time()

    q_emb = model.encode([q.question])
    D, I = index.search(np.array(q_emb), k=3)

    results = [documents[i] for i in I[0]]

    context = " ".join(results)

    answer = f"Answer based on context: {context[:300]}..."

    end = time.time()

    return {
        "answer": answer,
        "latency": end - start
    }
