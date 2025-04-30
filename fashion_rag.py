# fashion_rag.py

from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer
import chromadb

class FashionRAGAssistant:
    def __init__(self, collection_name, model_name="all-MiniLM-L6-v2"):
        # Initialize SentenceTransformer model
        self.model = SentenceTransformer(model_name)
        
        # Initialize Chroma collection client
        self.chroma_client = chromadb.PersistentClient(path="./chromadb_data")
        self.collection = self.chroma_client.get_collection(collection_name)
        
        # Check if the collection exists
        if self.collection is None:
            raise ValueError(f"Collection '{collection_name}' not found!")
        
        # OpenAI client setup
        self.client_oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def retrieve_context_with_rank(self, query, top_k=5):
        query_embedding = self.model.encode([query])[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        # Extract data
        documents = results["documents"][0]
        ids = results["ids"][0]
        distances = results["distances"][0]

        # Zip and sort by distance (ascending)
        ranked = sorted(zip(distances, ids, documents), key=lambda x: x[0])

        # Extract only documents in ranked order
        ranked_documents = [doc for _, _, doc in ranked]

        # Join documents as context
        context = "\n\n".join(ranked_documents)
        return context
    
    def generate_answer(self, query, context_docs):
        context = "\n\n".join(context_docs)
        prompt = f"""You are a fashion expert assistant. Given the context, answer the query.

Context:
{context}

Query:
{query}

Answer:"""
        response = self.client_oai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
