# Special Issue — PDF Question Answering System

A Retrieval-Augmented Generation (RAG) based question-answering system for interacting with the content of PDF documents.

The system retrieves the most relevant sections of a PDF document and uses a language model to generate answers based on the retrieved information.

## Overview

This project demonstrates a simple implementation of a Retrieval-Augmented Generation (RAG) pipeline for document-based question answering.

Instead of relying only on the language model's general knowledge, the system first searches the provided PDF for relevant information and then uses the retrieved content as context for generating the answer.

This helps the generated responses remain grounded in the source document.

## How It Works

The system follows these steps:

```text
PDF Document
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Embedding Generation
     ↓
FAISS Vector Database
     ↓
User Question
     ↓
Similarity Search
     ↓
Relevant Document Chunks
     ↓
Language Model
     ↓
Generated Answer
```

## Features

- PDF document loading and text extraction
- Text splitting into smaller chunks
- Multilingual semantic embeddings
- FAISS-based vector similarity search
- Retrieval of relevant document sections
- Persian question answering
- Context-grounded responses
- Simple command-line interface

## Technologies

- Python
- LangChain
- Hugging Face Sentence Transformers
- FAISS
- PyPDFium2
- OpenAI-compatible API

## Embedding Model

The project uses:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

This multilingual embedding model is used to convert document chunks and user queries into numerical vectors for semantic similarity search.

## Text Chunking

The PDF content is divided into smaller overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.

The current configuration is:

```python
chunk_size=1000
chunk_overlap=150
```

The overlap helps preserve contextual information between consecutive chunks.

## Retrieval

FAISS is used to store the document embeddings and perform similarity search.

For each user question, the system retrieves the five most relevant chunks:

```python
related_chunks = db.similarity_search(query, k=5)
```

The retrieved chunks are combined and provided to the language model as the source context.

## Answer Generation

The language model is instructed to answer the user's question using only the retrieved information from the document.

The system is designed to:

1. Use only the provided source context.
2. Avoid guessing or adding information from general knowledge.
3. Generate clear and accurate answers in Persian.
4. Indicate when the required information cannot be found in the document.

If the required information is not available in the retrieved context, the system responds with:

```text
اطلاعات کافی در فایل مورد نظر برای پاسخ به این سوال یافت نشد.
```

## Installation

Make sure Python is installed on your system.

Install the required dependencies:

```bash
pip install langchain-community
pip install langchain-text-splitters
pip install langchain-huggingface
pip install langchain-openai
pip install faiss-cpu
pip install pypdfium2
```

## Configuration

Before running the program, set the PDF file path in the source code:

```python
pdf_path = r"YOUR FILE ADDRESS"
```

Replace `YOUR FILE ADDRESS` with the path to the PDF document you want to use.

The API key is also configured directly in the source code:

```python
api_key="your-api-key"
```

Replace the placeholder with your API key before running the program.

> Do not publish a real API key in a public GitHub repository.

## Running the Project

Run the Python file:

```bash
python main.py
```

The program will load the PDF, process its content, create the vector database, and then ask for a question:

```text
چطور میتونم کمکتون کنم؟
```

Enter your question and the system will retrieve the most relevant sections of the document and generate an answer.

## Example

### Question

```text
دانشگاه صنعتی شیراز برای هر رشته چه ظرفیتی دارد؟
```

### Process

The system:

1. Loads the PDF document.
2. Extracts its text.
3. Splits the text into smaller chunks.
4. Generates embeddings for the chunks.
5. Creates a FAISS vector database.
6. Receives the user's question.
7. Retrieves the most relevant document chunks.
8. Provides the retrieved content to the language model.
9. Generates a Persian answer based on the retrieved information.

## Project Structure

```text
special-issue/
│
├── main.py
├── README.md
└── .gitignore
```

## Limitations

This project is a lightweight RAG implementation designed for educational and demonstration purposes.

Current limitations include:

- The vector database is rebuilt each time the program runs.
- The current implementation processes a single PDF document.
- Retrieval quality depends on the document structure and embedding model.
- The project currently uses a command-line interface.
- The generated answer depends on the quality and relevance of the retrieved context.

## Possible Future Improvements

The system can be extended with:

- Persistent vector database storage
- Support for multiple documents
- Document upload functionality
- Improved retrieval and reranking
- Hybrid search
- Conversation history
- REST API integration
- Django-based backend
- Docker support
- Source references and page-level citations
- A web-based user interface

## Purpose

This project demonstrates a practical application of Retrieval-Augmented Generation for question answering over domain-specific PDF documents.

It was developed as an AI-related project for the Special Issue of the Scientific Journal of Shiraz University of Technology.

## License

This project is provided for educational and research purposes.