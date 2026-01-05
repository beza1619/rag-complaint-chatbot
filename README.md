# RAG-Powered Complaint Analysis Chatbot for Financial Services

## 🏦 Project Overview
This project implements an intelligent complaint analysis system for CrediTrust Financial using Retrieval-Augmented Generation (RAG).

## 🎯 Business Problem
CrediTrust Financial receives thousands of customer complaints monthly across:
- Credit Cards
- Personal Loans
- Savings Accounts
- Money Transfers

## 📊 Key Features
- **Semantic Search**: Find relevant complaints using vector embeddings
- **Multi-Product Analysis**: Compare issues across financial products
- **Evidence-Based Answers**: Every answer cites source complaint excerpts
- **Non-Technical Interface**: Gradio web UI for business users

## 🛠️ Technical Implementation
1. **Data Processing**: Filtered CFPB complaints for 4 target products
2. **Text Chunking**: 500-character chunks with 50-character overlap
3. **Embeddings**: `all-MiniLM-L6-v2` model (384 dimensions)
4. **Vector Store**: ChromaDB with similarity search
5. **RAG Pipeline**: Retrieve -> Analyze -> Generate answers
6. **Interface**: Gradio web application

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 📁 Project Structure
```
rag-complaint-chatbot/
├── data/                   # Complaint datasets
├── vector_store/           # Embeddings and vector store
├── notebooks/              # EDA and development
├── app.py                 # Gradio interface
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 👤 Developer
**Bezawit Wondimneh** (GitHub: beza1619)

## 📅 Submission
- **Interim**: 04 Jan 2026
- **Final**: 13 Jan 2026
