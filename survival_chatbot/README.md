# Bear Grylls AI Survival Chatbot

A conversational AI chatbot powered by RAG (Retrieval-Augmented Generation) that provides expert survival advice based on US Army Field Manuals.

## Features

- Conversational memory - remembers your conversation context
- GPU-accelerated inference (Mistral 7B via Ollama)
- Retrieval from 881 pages of survival manuals
-  Clean, professional Gradio UI
- Optimal configuration: Page chunking + Nomic embeddings + Dense retrieval

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running with Mistral 7B model:
```bash
ollama pull mistral:7b
```

3. Run the chatbot:
```bash
python chatbot.py
```

4. Open your browser and go to: `http://127.0.0.1:7860`

## Usage

Simply type your survival questions into the chat interface. The chatbot will:
- Remember your previous questions and answers
- Retrieve relevant information from survival manuals
- Generate natural, conversational responses
- Format answers with proper structure and bullet points

Click "Clear Conversation" to start a fresh conversation.

## Technical Details

- **LLM**: Mistral 7B (via Ollama)
- **Embeddings**: Nomic-AI v1.5 (768 dimensions)
- **Vector Store**: ChromaDB with L2 distance
- **Chunking**: Page-based (881 chunks)
- **Retrieval**: Dense retrieval with cosine similarity
- **Top-K**: 5 most relevant chunks per query
- **Memory**: Last 6 conversation turns
