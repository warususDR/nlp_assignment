import gradio as gr
import json
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

DATA_DIR = Path('../processed_data')
VECTOR_STORE_DIR = Path('../vector_stores')

class SurvivalChatbot:
    def __init__(self):
        self.load_resources()
        self.conversation_history = []
        
    def load_resources(self):
        with open(DATA_DIR / 'chunks_page.json', 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        self.embedding_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device='cuda')
        
        chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR / 'chroma_db'))
        self.chroma_collection = chroma_client.get_collection(name='survival_nomic_page')
        
        self.llm_model = 'mistral:7b'
        
    def retrieve_context(self, query, top_k=5):
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['documents', 'embeddings']
        )
        
        retrieved_docs = []
        for i, (doc_id, doc, stored_embedding) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['embeddings'][0]
        )):
            stored_emb_array = np.array(stored_embedding)
            
            dot_product = np.dot(query_embedding, stored_emb_array)
            query_norm = np.linalg.norm(query_embedding)
            stored_norm = np.linalg.norm(stored_emb_array)
            
            cosine_sim = dot_product / (query_norm * stored_norm)
            similarity = float(np.clip(cosine_sim, 0.0, 1.0))
            
            retrieved_docs.append({
                'text': doc,
                'score': similarity
            })
        
        return retrieved_docs
    
    def format_conversation_context(self):
        if not self.conversation_history:
            return ""
        
        context = "\n\nPrevious conversation:\n"
        for entry in self.conversation_history[-6:]:
            context += f"User: {entry['user']}\n"
            context += f"You: {entry['assistant']}\n\n"
        return context
    
    def chat(self, user_message):
        retrieved_docs = self.retrieve_context(user_message, top_k=5)
        
        context = "\n\n".join([f"[{i+1}] {doc['text'][:400]}" for i, doc in enumerate(retrieved_docs)])
        
        conversation_context = self.format_conversation_context()
        
        system_prompt = """You are a famous survival expert Bear Grylls. 
                    Provide practical, actionable advice based on the provided context. 
                    Try to act natural as a human would and rephrase the context if necessary.
                    Do not mention you are using the context, pretend it's your own knowledge.
                    Keep the answer short up to 150 words.
                    
                    If the user asks about survival, wilderness, emergencies, or outdoor situations, use the context provided.
                    If the user is just chatting about unrelated topics, respond naturally without using the context.
                 """
        
        user_prompt = f"""Context from survival manuals:
{context}
{conversation_context}
Question: {user_message}

Answer:"""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 500
                }
            )
            answer = response['message']['content'].strip()
        except Exception as e:
            answer = f"I apologize, but I encountered an error: {str(e)}"
        
        self.conversation_history.append({
            'user': user_message,
            'assistant': answer
        })
        
        return answer
    
    def reset_conversation(self):
        self.conversation_history = []
        return "Conversation history cleared. Let's start fresh!"

chatbot = SurvivalChatbot()

def chat_interface(message, history):
    response = chatbot.chat(message)
    return response

def reset_interface():
    chatbot.reset_conversation()
    return []

with gr.Blocks(title="Bear Grylls Survival Expert", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Bear Grylls AI 
        
        I'm Bear Grylls, and I've spent years mastering survival in the harshest environments on Earth. 
        I'm here to help you with any survival situation.
        
        Ask me about water procurement, shelter building, fire starting, navigation, first aid, or handling dangerous wildlife!
        """
    )
    
    chatbot_interface = gr.ChatInterface(
        fn=chat_interface,
        examples=[
            "How do I find water in the desert?",
            "What's the best way to build a shelter in cold weather?",
            "How can I start a fire without matches?",
        ],
        title="",
        description="",
        retry_btn=None,
        undo_btn=None,
        clear_btn="🔄 Clear Conversation",
        chatbot=gr.Chatbot(height=400)
    )
    
    gr.Markdown(
        """
        ---
        **Powered by:** Mistral 7B • Nomic Embeddings • ChromaDB • Page-based Chunking
        
        *By Dmytrii Rakitenko*
        """
    )

if __name__ == "__main__":
    print("Loading Bear Grylls Survival Expert Chatbot...")
    print("This may take a moment to load models onto GPU...")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
