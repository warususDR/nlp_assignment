# RAG-Based Survival Chatbot

Comprehensive evaluation and implementation of a Retrieval-Augmented Generation system for military survival manuals with conversational AI interface.

## Overview

This project evaluates 18 RAG configurations across chunking strategies, retrieval methods, embedding models, and LLMs using US Army Field Manuals FM 21-76 and FM 3-05-70 (881 pages). The optimal configuration powers a production-ready chatbot with conversational memory.

## Dataset

- **FM 21-76**: Survival (229 pages)
- **FM 3-05-70**: Survival Manual (652 pages)
- **Total**: 881 pages of military survival doctrine

## Evaluation Results

**180 evaluations** across:
- **Chunking**: page (881), fixed_size (1540), sentence (1577)
- **Retrieval**: TF-IDF, dense_nomic, dense_minilm
- **Embeddings**: Nomic (768d), MiniLM (384d)
- **LLMs**: Gemma 3 4B, Mistral 7B

**Optimal Configuration**:
- Chunking: page
- Retrieval: dense_nomic (cosine similarity: 0.77)
- LLM: Mistral 7B
- Combined Score: 0.6 × retrieval + 0.4 × context_overlap

## Project Structure

```
nlp_assignment/
├── rag_pipeline_and_evaluation.ipynb    # Complete evaluation pipeline
├── processed_data/                       # Chunked documents (3 strategies)
├── vector_stores/                        # ChromaDB collections (6 total)
├── evaluation_results/                   # CSV, JSON, visualizations
└── survival_chatbot/                     # Production chatbot application
    ├── chatbot.py                        # Gradio interface with memory
    ├── requirements.txt
    └── README.md
```

## Technical Stack

- **Embeddings**: sentence-transformers (Nomic, MiniLM)
- **Vector Store**: ChromaDB (persistent)
- **LLM**: Ollama (Mistral 7B, Gemma 3 4B)
- **Retrieval**: Manual cosine similarity calculation
- **Interface**: Gradio 4.44.0
- **Hardware**: CUDA-accelerated (RTX 4060)

## Installation

```powershell
pip install sentence-transformers chromadb ollama gradio scikit-learn pandas matplotlib seaborn jupyter
```

Install Ollama and pull models:
```powershell
ollama pull mistral:7b
ollama pull gemma2:9b
```

## Usage

### Run Evaluation

```powershell
jupyter notebook rag_pipeline_and_evaluation.ipynb
```

### Launch Chatbot

```powershell
cd survival_chatbot
python chatbot.py
```

Access at `http://127.0.0.1:7860`

## Key Findings

1. **Page chunking outperforms** fixed-size and sentence-based strategies
2. **Nomic embeddings** (unnormalized, 768d) achieve highest retrieval scores with manual cosine similarity
3. **Mistral 7B** provides better context utilization than Gemma 3 4B
4. **Dense retrieval** (0.77 avg) significantly outperforms TF-IDF (0.19 avg)
5. **Conversational memory** (6 turns) enables natural multi-turn dialogue

## Evaluation Metrics

- **Retrieval Score**: Cosine similarity between query and retrieved documents
- **Context Overlap**: Jaccard similarity with ground truth context
- **Generation Time**: LLM response latency
- **Combined Score**: Weighted average (60% retrieval, 40% overlap)

## Chatbot Features

- Conversational memory (last 6 turns)
- GPU-accelerated inference
- Optimal configuration from evaluation
- Professional Gradio interface
- Context-aware responses from 881 pages

## Author

Dmytrii Rakitenko
