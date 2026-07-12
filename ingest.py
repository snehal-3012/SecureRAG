import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
DB_DIR = "faiss_index"


def ingest_documents():
    print("Looking for PDF files in the 'data' folder...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(
            f"Created '{DATA_DIR}' folder. Please place a PDF inside it and run this script again."
        )
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{DATA_DIR}'. Please add one and try again.")
        return

    # 1. Load the first PDF found
    file_path = os.path.join(DATA_DIR, pdf_files[0])
    print(f"Loading document: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split text into manageable chunks
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create embeddings and store in FAISS vector database
    print("Creating mathematical embeddings (this may take a moment on CPU)...")
    # We use a lightweight open-source embedding model from Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. Save the FAISS index locally so we don't have to re-process the PDF
    vectorstore.save_local(DB_DIR)
    print(f"Success! Vector database saved to '{DB_DIR}'. You can now run query.py")


if __name__ == "__main__":
    ingest_documents()
