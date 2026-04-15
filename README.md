# RAG-Based Question Answering System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system using FastAPI. Users can upload documents and ask questions based on the content.

## Features
- Upload PDF and TXT documents
- Chunk and embed text using sentence-transformers
- Store embeddings using FAISS
- Retrieve relevant chunks using similarity search
- Generate answers based on retrieved content

## Tech Stack
- FastAPI
- Sentence Transformers
- FAISS
- Python

## How to Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
