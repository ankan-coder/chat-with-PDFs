# Chat with PDF - RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system that allows you to chat with PDF documents using Google's Gemini AI. This system intelligently stores embeddings to reduce API costs and supports multiple PDFs in storage.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Cost Optimization](#cost-optimization)
- [Technical Details](#technical-details)

## 🎯 Overview

This RAG (Retrieval-Augmented Generation) system enables natural language conversations with PDF documents. It uses vector embeddings to find relevant content from your PDFs and generates accurate answers using Google's Gemini AI models.

**Key Capabilities:**
- Extract and process text from PDF files
- Create vector embeddings for semantic search
- Store embeddings persistently to avoid re-computation
- Support multiple PDFs simultaneously
- Intelligent caching to reduce API costs
- Interactive chat interface

## ✨ Features

### Core Features
- **PDF Text Extraction**: Extracts text from all pages of PDF documents
- **Text Chunking**: Splits large documents into manageable chunks (800 words by default)
- **Vector Embeddings**: Creates embeddings using Google's `text-embedding-004` model
- **Semantic Search**: Uses cosine similarity to find most relevant document chunks
- **AI-Powered Answers**: Generates answers using Gemini 2.5 Flash model
- **Multiple PDF Support**: Switch between different PDFs seamlessly

### Cost Optimization Features
- **Persistent Embedding Storage**: Saves embeddings to disk in JSON format
- **Smart Caching**: Only re-embeds PDFs when they've been modified
- **Metadata Tracking**: Tracks creation date, update date, and modification time
- **Automatic Detection**: Detects PDF changes and re-embeds only when necessary

### User Experience Features
- **Interactive Selection**: Choose PDFs by number or filename
- **Embedding Status**: View all saved embeddings with metadata
- **Continuous Chat**: Ask multiple questions in a session
- **Clear Feedback**: Shows when embeddings are loaded vs. created

## 🔄 How It Works

### RAG Pipeline

1. **Document Processing**
   - PDF text extraction using `pypdf`
   - Text chunking into 800-word segments
   - Creation of vector embeddings for each chunk

2. **Embedding Storage**
   - Embeddings saved to `embeddings/` directory as JSON files
   - Metadata includes: PDF name, creation date, update date, modification time
   - Automatic detection of PDF changes

3. **Query Processing**
   - User question converted to vector embedding
   - Cosine similarity search finds top 3 most relevant chunks
   - Context assembled from relevant chunks

4. **Answer Generation**
   - Context and question sent to Gemini 2.5 Flash
   - LLM generates answer based on retrieved context
   - Answer returned to user

### Workflow Diagram

```
PDF File
   ↓
Text Extraction
   ↓
Text Chunking (800 words/chunk)
   ↓
Check if embeddings exist
   ├─ Yes → Check modification time
   │   ├─ Modified → Re-embed
   │   └─ Not Modified → Load saved embeddings
   └─ No → Create embeddings
   ↓
Save embeddings with metadata
   ↓
User Question
   ↓
Question Embedding
   ↓
Similarity Search (Cosine)
   ↓
Top 3 Chunks Retrieved
   ↓
Context + Question → LLM
   ↓
Answer Generated
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Google Gemini API key

### Install Dependencies

```bash
pip install pypdf google-genai python-dotenv scikit-learn numpy
```

Or create a `requirements.txt` file:

```txt
pypdf
google-genai
python-dotenv
scikit-learn
numpy
```

Then install:

```bash
pip install -r requirements.txt
```

## ⚙️ Setup

1. **Get Google Gemini API Key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the API key

2. **Create Environment File**
   - Create a `.env` file in the project root
   - Add your API key:

   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Add PDF Files**
   - Place your PDF files in the project root directory
   - The system will automatically detect all `.pdf` files

## 📖 Usage

### Basic Usage

1. **Run the Application**
   ```bash
   python app.py
   ```

2. **Select a PDF**
   - The system will show all available PDFs
   - Enter the PDF number (e.g., `1`) or filename
   - Press Enter

3. **First Time Processing**
   - If embeddings don't exist, the system will:
     - Extract text from PDF
     - Create chunks
     - Generate embeddings (may take a moment)
     - Save embeddings to `embeddings/` directory

4. **Subsequent Runs**
   - If embeddings exist and PDF hasn't changed:
     - Loads saved embeddings instantly
     - No API calls for embedding generation
     - Ready to chat immediately

5. **Chat with PDF**
   - Type your question and press Enter
   - Get AI-generated answers based on PDF content
   - Type `quit`, `exit`, or `q` to stop

### Example Session

```
============================================================
PDF Chat with RAG - Multiple PDF Support
============================================================

Available PDFs:
  1. document1.pdf
  2. document2.pdf

Saved Embeddings:
  1. document1.pdf (Created: 2024-01-15T10:30:00, Updated: 2024-01-15T10:30:00, Chunks: 45)

------------------------------------------------------------
Enter PDF number (1-2) or PDF filename: 1

Selected PDF: document1.pdf
Loading saved embeddings...
Loaded 45 chunks from saved embeddings
Embeddings created: 2024-01-15T10:30:00
Last updated: 2024-01-15T10:30:00

============================================================
Chat with your PDF! (Type 'quit' or 'exit' to stop)
============================================================

Ask a question: What is the main topic of this document?

------------------------------------------------------------
Answer:
The main topic of this document is...
------------------------------------------------------------
```

## 🏗️ Architecture

### Components

1. **Text Processing Module**
   - `chunk_text()`: Splits text into chunks
   - `process_pdf()`: Handles PDF extraction and processing

2. **Embedding Management**
   - `save_embeddings()`: Saves embeddings with metadata
   - `load_embeddings()`: Loads saved embeddings
   - `needs_reembedding()`: Checks if re-embedding is needed

3. **PDF Management**
   - `get_pdf_metadata()`: Extracts PDF file metadata
   - `list_available_pdfs()`: Lists all PDFs in directory
   - `list_saved_embeddings()`: Lists all saved embeddings

4. **RAG Pipeline**
   - Question embedding generation
   - Cosine similarity search
   - Context retrieval
   - LLM answer generation

### Data Flow

```
PDF → Text → Chunks → Embeddings → Storage (JSON)
                                    ↓
User Question → Embedding → Similarity Search → Top Chunks
                                    ↓
                            Context + Question → LLM → Answer
```

## 📁 File Structure

```
Chat with PDF - RAG/
│
├── app.py                 # Main application file
├── .env                   # Environment variables (API key)
├── README.md              # This file
├── requirements.txt       # Python dependencies (optional)
│
├── embeddings/           # Generated embeddings directory
│   ├── document1_pdf.json
│   ├── document2_pdf.json
│   └── ...
│
└── *.pdf                 # Your PDF files
```

### Embedding File Structure

Each embedding file (`embeddings/*.json`) contains:

```json
{
  "pdf_name": "document.pdf",
  "pdf_path": "/absolute/path/to/document.pdf",
  "created_at": "2024-01-15T10:30:00.123456",
  "updated_at": "2024-01-15T10:30:00.123456",
  "modified_time": 1705312200.123,
  "num_chunks": 45,
  "documents": [
    {
      "text": "chunk text content...",
      "embedding": [0.123, 0.456, ...]
    },
    ...
  ]
}
```

## 💰 Cost Optimization

### How Costs Are Reduced

1. **One-Time Embedding**
   - Embeddings created once per PDF
   - Stored locally in JSON format
   - No re-embedding unless PDF changes

2. **Smart Caching**
   - Modification time tracking
   - Automatic detection of PDF updates
   - Re-embedding only when necessary

3. **Efficient Storage**
   - JSON format for easy access
   - No database overhead
   - Fast loading times

### Cost Comparison

**Without Caching:**
- Every run: Embedding API calls for all chunks
- Example: 100 chunks × $0.0001 = $0.01 per run
- 10 runs = $0.10

**With Caching:**
- First run: $0.01 (create embeddings)
- Subsequent runs: $0.00 (load from disk)
- 10 runs = $0.01 (90% savings!)

## 🔧 Technical Details

### Models Used

- **Embedding Model**: `text-embedding-004` (Google Gemini)
- **LLM Model**: `gemini-2.5-flash` (Google Gemini)

### Parameters

- **Chunk Size**: 800 words (configurable in `chunk_text()`)
- **Top Chunks**: 3 most similar chunks retrieved
- **Similarity Metric**: Cosine similarity

### Dependencies

- `pypdf`: PDF text extraction
- `google-genai`: Google Gemini API client
- `python-dotenv`: Environment variable management
- `scikit-learn`: Cosine similarity calculation
- `numpy`: Numerical operations

### Performance Considerations

- **Embedding Creation**: ~1-5 seconds per 100 chunks (depends on API)
- **Embedding Loading**: <1 second (from local JSON)
- **Similarity Search**: <1 second for typical documents
- **Answer Generation**: ~2-5 seconds (depends on API)

## 🛠️ Customization

### Change Chunk Size

Edit the `chunk_text()` function:

```python
def chunk_text(text, chunk_size=800):  # Change 800 to desired size
    # ...
```

### Change Number of Top Chunks

Edit the similarity search section:

```python
top_chunks = [chunk for _, chunk in best_chunks[:3]]  # Change 3 to desired number
```

### Change LLM Model

Edit the model name:

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",  # Change to desired model
    contents=prompt
)
```

## 📝 Notes

- Embeddings are stored in the `embeddings/` directory
- Each PDF gets its own JSON file based on filename
- The system automatically creates the `embeddings/` directory
- PDF modification time is checked to determine if re-embedding is needed
- The `created_at` timestamp is preserved when updating embeddings

## 🐛 Troubleshooting

### Issue: "No PDF files found"
- **Solution**: Ensure PDF files are in the same directory as `app.py`

### Issue: "API key not found"
- **Solution**: Check that `.env` file exists and contains `GEMINI_API_KEY=your_key`

### Issue: Embeddings not loading
- **Solution**: Check that `embeddings/` directory exists and contains JSON files
- **Solution**: Verify JSON files are not corrupted

### Issue: Slow embedding creation
- **Solution**: This is normal for large PDFs. The system will cache results for future use.

## 📄 License

This project is open source and available for personal and educational use.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

---

**Happy Chatting with Your PDFs! 📚💬**
#
