# SecureRAG

A fully offline, local Retrieval-Augmented Generation (RAG) system built with LangChain, FAISS, and Ollama. This project allows you to upload any PDF document and ask an AI questions about its contents, all while keeping your data 100% private and running on your local CPU.

## Features

- **100% Offline & Private:** Uses Ollama to run large language models (like Meta's Llama 3) entirely locally. Your documents are never sent to the cloud.
- **Modern LangChain Architecture:** Built using the latest LangChain Expression Language (LCEL) for highly efficient, modular chaining.
- **Vector Search (FAISS):** Embeds text chunks mathematically using Hugging Face's `sentence-transformers` and stores them in a FAISS vector database for lightning-fast semantic retrieval.
- **Automated PDF Parsing:** Uses `PyPDFLoader` to seamlessly extract text from research papers, manuals, or articles.

## Tech Stack

- **Framework:** [LangChain](https://www.langchain.com/) (LCEL)
- **Local LLM Engine:** [Ollama](https://ollama.com/) (running `llama3`)
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
- **Embeddings:** Hugging Face `all-MiniLM-L6-v2` (via `sentence-transformers`)

## Installation

### 1. Install Ollama & The LLM
First, you need to install the AI engine that will answer your questions.
Install Ollama from [ollama.com](https://ollama.com/) (or run `curl -fsSL https://ollama.com/install.sh | sh` on Linux/macOS).
Then, download the Llama 3 model:
```bash
ollama pull llama3
```

### 2. Setup the Python Environment
We highly recommend using Conda to avoid dependency conflicts.
```bash
conda create -n rag_qa python=3.10 -y
conda activate rag_qa
```

To ensure compatibility and prevent GPU driver issues, install the CPU version of PyTorch and a stable version of Numpy before installing the rest of the requirements:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2.0.0"
pip install -r requirements.txt
```

## How to Use

### Step 1: Ingesting a PDF
1. Run the ingest script to create the data folder:
   ```bash
   python ingest.py
   ```
2. Place any `.pdf` file inside the newly created `data/` folder.
3. Run the ingest script again. It will chunk the text, create mathematical embeddings, and save a FAISS vector database to your disk:
   ```bash
   python ingest.py
   ```

### Step 2: Chatting with your PDF
Once the database is created, you can start the interactive CLI chat:
```bash
python query.py
```
Type your question, and Llama 3 will read the relevant sections of your PDF to give you a precise answer!

## Architecture Flow
1. **Document -> Chunks:** The PDF is loaded and split into 500-character chunks with a 50-character overlap.
2. **Chunks -> FAISS:** The chunks are converted to vectors and stored in a local FAISS index.
3. **Question -> FAISS:** The user asks a question, which is converted to a vector to find the top 3 most relevant chunks in the database.
4. **Chunks + Question -> Llama 3:** The relevant text and the user's question are injected into a prompt and sent to the local Llama 3 model, which generates a concise, accurate answer.
