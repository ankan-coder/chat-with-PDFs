from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# Directory to store embeddings
EMBEDDINGS_DIR = "embeddings"

def chunk_text(text, chunk_size=800):
    words = text.split() # Turn text string to a list of substrings, eg., text = "I am Ankan", text.split = ["I", "am", "Ankan"]
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def get_pdf_metadata(pdf_path):
    """Get PDF file metadata (name, modification time)"""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        return None
    
    return {
        "name": pdf_file.name,
        "path": str(pdf_file.absolute()),
        "modified_time": os.path.getmtime(pdf_path)
    }

def get_embedding_file_path(pdf_name):
    """Get the path to the embedding file for a PDF"""
    # Create embeddings directory if it doesn't exist
    Path(EMBEDDINGS_DIR).mkdir(exist_ok=True)
    
    # Sanitize PDF name for filename
    safe_name = "".join(c for c in pdf_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_')
    return os.path.join(EMBEDDINGS_DIR, f"{safe_name}.json")

def save_embeddings(pdf_metadata, documents):
    """Save embeddings with metadata to a JSON file"""
    embedding_file = get_embedding_file_path(pdf_metadata["name"])
    
    # Check if embeddings already exist to preserve created_at
    existing_data = None
    if os.path.exists(embedding_file):
        try:
            with open(embedding_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except:
            pass
    
    created_at = existing_data.get("created_at", datetime.now().isoformat()) if existing_data else datetime.now().isoformat()
    
    data = {
        "pdf_name": pdf_metadata["name"],
        "pdf_path": pdf_metadata["path"],
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(),
        "modified_time": pdf_metadata["modified_time"],
        "num_chunks": len(documents),
        "documents": documents
    }
    
    with open(embedding_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Embeddings saved to {embedding_file}")
    return embedding_file

def load_embeddings(pdf_name):
    """Load embeddings from a JSON file"""
    embedding_file = get_embedding_file_path(pdf_name)
    
    if not os.path.exists(embedding_file):
        return None
    
    with open(embedding_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def needs_reembedding(pdf_path, saved_data):
    """Check if PDF needs re-embedding based on modification time"""
    if saved_data is None:
        return True
    
    current_modified_time = os.path.getmtime(pdf_path)
    saved_modified_time = saved_data.get("modified_time", 0)
    
    return current_modified_time > saved_modified_time

def list_available_pdfs():
    """List all PDFs in the current directory"""
    pdf_files = list(Path('.').glob('*.pdf'))
    return [str(pdf) for pdf in pdf_files]

def list_saved_embeddings():
    """List all saved embeddings"""
    if not os.path.exists(EMBEDDINGS_DIR):
        return []
    
    embedding_files = list(Path(EMBEDDINGS_DIR).glob('*.json'))
    saved_pdfs = []
    
    for emb_file in embedding_files:
        try:
            with open(emb_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                saved_pdfs.append({
                    "name": data.get("pdf_name", emb_file.stem),
                    "created_at": data.get("created_at", "Unknown"),
                    "updated_at": data.get("updated_at", "Unknown"),
                    "num_chunks": data.get("num_chunks", 0)
                })
        except Exception as e:
            print(f"Error reading {emb_file}: {e}")
    
    return saved_pdfs

def process_pdf(pdf_path, client):
    """Process a PDF: extract text, chunk, and create embeddings"""
    print(f"\nProcessing PDF: {pdf_path}")
    
    # Step 1 - Extract Text from PDF
    pdf = PdfReader(pdf_path)
    text = ""
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    
    # Step 2 - Chunk the text
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    
    # Step 3 - Convert chunks to embeddings
    print("Creating embeddings... (this may take a moment)")
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=chunks
    )
    
    embeddings = [e.values for e in response.embeddings]
    
    documents = []
    for chunk, embedding in zip(chunks, embeddings):
        documents.append({
            "text": chunk,
            "embedding": embedding
        })
    
    # Get PDF metadata and save embeddings
    pdf_metadata = get_pdf_metadata(pdf_path)
    save_embeddings(pdf_metadata, documents)
    
    return documents


if __name__ == "__main__":
    load_dotenv() # .env file loads
    
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    # Show available PDFs
    available_pdfs = list_available_pdfs()
    saved_embeddings = list_saved_embeddings()
    
    print("=" * 60)
    print("PDF Chat with RAG - Multiple PDF Support")
    print("=" * 60)
    
    if available_pdfs:
        print("\nAvailable PDFs:")
        for i, pdf in enumerate(available_pdfs, 1):
            print(f"  {i}. {pdf}")
    else:
        print("\nNo PDF files found in the current directory.")
        exit(1)
    
    if saved_embeddings:
        print("\nSaved Embeddings:")
        for i, emb in enumerate(saved_embeddings, 1):
            print(f"  {i}. {emb['name']} (Created: {emb['created_at'][:19]}, Updated: {emb['updated_at'][:19]}, Chunks: {emb['num_chunks']})")
    
    # Select PDF
    print("\n" + "-" * 60)
    pdf_choice = input(f"Enter PDF number (1-{len(available_pdfs)}) or PDF filename: ").strip()
    
    # Try to parse as number first
    try:
        pdf_index = int(pdf_choice) - 1
        if 0 <= pdf_index < len(available_pdfs):
            selected_pdf = available_pdfs[pdf_index]
        else:
            print("Invalid number. Using first PDF.")
            selected_pdf = available_pdfs[0]
    except ValueError:
        # Try to find by filename
        if pdf_choice in available_pdfs:
            selected_pdf = pdf_choice
        elif os.path.exists(pdf_choice):
            selected_pdf = pdf_choice
        else:
            print(f"PDF '{pdf_choice}' not found. Using first available PDF.")
            selected_pdf = available_pdfs[0]
    
    print(f"\nSelected PDF: {selected_pdf}")
    
    # Check if embeddings exist and if they need updating
    pdf_metadata = get_pdf_metadata(selected_pdf)
    saved_data = load_embeddings(pdf_metadata["name"])
    
    if saved_data and not needs_reembedding(selected_pdf, saved_data):
        print("Loading saved embeddings...")
        documents = saved_data["documents"]
        print(f"Loaded {len(documents)} chunks from saved embeddings")
        print(f"Embeddings created: {saved_data.get('created_at', 'Unknown')}")
        print(f"Last updated: {saved_data.get('updated_at', 'Unknown')}")
    else:
        if saved_data:
            print("PDF has been modified. Re-embedding...")
        else:
            print("No saved embeddings found. Creating new embeddings...")
        
        documents = process_pdf(selected_pdf, client)
        print(f"Total embeddings created: {len(documents)}")
    
    # Chat loop
    print("\n" + "=" * 60)
    print("Chat with your PDF! (Type 'quit' or 'exit' to stop)")
    print("=" * 60)
    
    while True:
        question = input("\nAsk a question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        # Convert question to vector embedding
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=[question]
        )
        
        question_embedding = response.embeddings[0].values
        
        # Similarity Search (Core logic)
        question_embedding = np.array(question_embedding).reshape(1, -1)
        best_chunks = []
        for doc in documents:
            doc_embedding = np.array(doc["embedding"]).reshape(1, -1)
            similarity = cosine_similarity(question_embedding, doc_embedding)[0][0]
            best_chunks.append((similarity, doc["text"]))
        
        best_chunks.sort(reverse=True)
        top_chunks = [chunk for _, chunk in best_chunks[:3]]
        
        # Send to LLM
        context = "\n\n".join(top_chunks)
        
        prompt = f"""Answer the question using only the context below.
        If the answer is not in the context, say you don't know.

        Context:
        {context}

        Question:
        {question}
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        print("\n" + "-" * 60)
        print("Answer:")
        print(response.text)
        print("-" * 60)


