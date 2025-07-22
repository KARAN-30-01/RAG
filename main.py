import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
import requests
import json
from typing import List, Dict
from transformers import pipeline
import torch

print("All libraries imported successfully!")

class SimpleDocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )

    def extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return text

    def extract_text_from_docx(self, file_path: str) -> str:
        text = ""
        try:
            doc = DocxDocument(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
        return text

    def extract_text_from_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]', ' ', text)
        return text.strip()

    def process_document(self, file_path: str) -> List[Dict]:
        if file_path.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            text = self.extract_text_from_docx(file_path)
        elif file_path.lower().endswith('.txt'):
            text = self.extract_text_from_txt(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            return []

        if not text:
            print(f"No text extracted from {file_path}")
            return []

        text = self.clean_text(text)

        filename = os.path.basename(file_path).lower()
        doc_type = "placement" if "placement" in filename else "si" if "si" in filename else "general"

        chunks = self.text_splitter.split_text(text)

        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                'content': chunk,
                'source': os.path.basename(file_path),
                'type': doc_type,
                'chunk_id': i
            })

        print(f"Processed {file_path}: {len(chunks)} chunks")
        return documents


class ColabLLMHandler:
    def __init__(self):
        self.local_model = None
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.hf_model = None

    def load_local_model(self, model_name="microsoft/DialoGPT-medium"): 
        try:
            print(f"Loading local model: {model_name}...")
            self.hf_model = pipeline( 
                "text-generation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                max_length=512, 
                do_sample=True, 
                temperature=0.7 
            )
            print("Local model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading local model: {e}")
            return False

    def generate_with_groq(self, prompt: str, api_key: str) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {api_key}", 
                "Content-Type": "application/json" 
            }

            data = {
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}], 
                "temperature": 0.7, 
                "max_tokens": 512 
            }

            response = requests.post(self.groq_url, headers=headers, json=data, timeout=30) 

            if response.status_code == 200: 
                return response.json()["choices"][0]["message"]["content"]


"""Groq (like OpenAI-style APIs) responds with a JSON like:

json
Copy
Edit
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Here’s your answer!"
      }
    }
  ]
}"""


            else:
                print(f"Groq API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Groq error: {e}")
            return None

    def generate_with_local_hf(self, prompt: str) -> str:
        try:
            if self.hf_model is None:
                return None

            if len(prompt) > 1000:
                prompt = prompt[:1000] + "..."

            result = self.hf_model(prompt, max_length=len(prompt.split()) + 100,
                                  num_return_sequences=1, pad_token_id=50256)

            generated_text = result[0]['generated_text']

            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()

            return response if response else None
        except Exception as e:
            print(f"Local HF model error: {e}")
            return None

    def generate_response(self, prompt: str, api_key: str = None, provider: str = "groq") -> str:
        if api_key:
            if provider == "groq":
                print("Using Groq API...")
                response = self.generate_with_groq(prompt, api_key)
                if response:
                    return response

        if self.hf_model:
            print("Using local Hugging Face model...")
            response = self.generate_with_local_hf(prompt)
            if response:
                return response

        return "Sorry, I couldn't generate a response."

class ColabPlacementRAG:
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.processor = SimpleDocumentProcessor()
        self.llm_handler = ColabLLMHandler()
        self.api_key = None
        self.provider = "groq"

    def initialize(self):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded successfully!")

    def load_documents(self, file_paths: List[str]):
        print(f"Processing {len(file_paths)} files...")
        self.documents = []

        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            docs = self.processor.process_document(file_path)
            self.documents.extend(docs)

        print(f"Total documents loaded: {len(self.documents)}")

    def create_index(self):
        if not self.documents:
            raise ValueError("No documents loaded!")

        print("Creating embeddings...")
        texts = [doc['content'] for doc in self.documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True) 

        dimension = embeddings.shape[1] 
        self.index = faiss.IndexFlatL2(dimension) 
        self.index.add(embeddings.astype('float32'))

        print("Vector index created successfully!")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if not self.index:
            return []

        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                results.append(doc)

        return results

    def answer_question(self, question: str) -> str:
        relevant_docs = self.search(question, k=5)

        if not relevant_docs:
            return "I couldn't find relevant information to answer your question."

        context = ""
        for doc in relevant_docs[:3]:
            context += f"Source: {doc['source']}\n{doc['content']}\n\n"

        prompt = f"""Based on the following context from placement documents, answer the question directly and concisely:

Context:
{context}

Question: {question}

Provide a helpful answer based on the context:"""

        response = self.llm_handler.generate_response(prompt, self.api_key, self.provider)

        return response

    def set_api_key(self, api_key: str, provider: str = "groq"):
        self.api_key = api_key.strip() if api_key else None
        self.provider = provider
        print(f"API key set for {provider.upper()}!")

    def load_local_model(self):
        success = self.llm_handler.load_local_model()
        if success:
            print("Local model loaded successfully!")
        else:
            print("Failed to load local model.")


class TerminalRAG:
    def __init__(self):
        self.rag = ColabPlacementRAG()
        self.rag.initialize()

    def display_menu(self):
        print("\n" + "="*60)
        print("         PLACEMENT & SI CHRONICLES RAG SYSTEM")
        print("="*60)
        print("1. Set API Key (Groq)")
        print("2. Load Local Model")
        print("3. Load Documents")
        print("4. Ask Questions")
        print("7. Exit")
        print("="*60)

    def set_api_key(self):
        print("\nAvailable providers:")
        print("Groq - Free API: https://console.groq.com/")

        provider = "groq"
        api_key = input(f"Enter your {provider.upper()} API key: ").strip()

        if api_key:
            self.rag.set_api_key(api_key, provider)
        else:
            print("No API key provided.")

    def load_local_model(self):
        print("\nLoading local model...")
        self.rag.load_local_model()

    def load_documents(self):

        file_paths = []


        print("\nEnter file paths (one per line, empty line to finish):")
        print("Supported formats: .pdf, .docx, .txt")

        while True:
            path = input("File path: ").strip()
            if not path: #enter empty path to stop loop
                break
            file_paths.append(path)

        if file_paths:
            print(f"\nFound {len(file_paths)} files to process...")
            self.rag.load_documents(file_paths)

            if self.rag.documents:
                self.rag.create_index()
                print("Documents loaded and indexed successfully!")
            else:
                print("No documents were successfully processed.")
        else:
            print("No files provided!")

    def ask_questions(self):
        if not self.rag.documents:
            print("\nNo documents loaded! Please load documents first.")
            return

        print("\nQuestion & Answer Mode")
        print("Type 'quit' to return to main menu")
        print("-" * 40)

        while True:
            question = input("\nYour question: ").strip()

            if question.lower() == 'quit':
                break

            if not question:
                print("Please enter a question.")
                continue

            print("\nProcessing...")
            answer = self.rag.answer_question(question)

            print("\nAnswer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)

    def run(self):
        print("Initializing RAG System...")

        while True:
            self.display_menu()
            choice = input("\nEnter your choice (1-7): ").strip()

            if choice == "1":
                self.set_api_key()
            elif choice == "2":
                self.load_local_model()
            elif choice == "3":
                self.load_documents()
            elif choice == "4":
                self.ask_questions()
            elif choice == "7":
                print("\nGoodbye!")
                break
            else:
                print("\nInvalid choice! Please enter a number between 1-7.")



if __name__ == "__main__":
    terminal_rag = TerminalRAG()
    terminal_rag.run()
