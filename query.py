import sys
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_DIR = "faiss_index"


def main():
    print("Loading vector database...")
    try:
        # We must load the DB with the exact same embedding model we used in ingest.py
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(
            DB_DIR, embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error loading FAISS index. Did you run ingest.py first? Error: {e}")
        return

    # Create the retriever (fetches top 3 most relevant text chunks from the DB)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Connect to the local Ollama LLM
    # Note: We are using "llama3" here. You must have it pulled in Ollama!
    llm = Ollama(model="llama3")

    # Define the system prompt
    system_prompt = (
        "You are an expert assistant for question-answering tasks. "
        "Use the following pieces of retrieved context from a document to answer the question. "
        "If the answer is not contained in the context, say that you don't know. "
        "Keep the answer concise and direct."
        "\n\nContext:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Helper function to combine the documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the RAG chain using modern LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n==================================")
    print("--- 🤖 SecureRAG System Ready ---")
    print("==================================")
    print("Type your question about the PDF below (or 'exit' to quit):")

    while True:
        question = input("\nQuestion: ")
        if question.lower() in ["exit", "quit"]:
            break

        print("Thinking...")
        # This sends the question to FAISS, grabs context, and passes it to Llama 3
        response = rag_chain.invoke(question)
        print(f"\nAnswer: {response}")


if __name__ == "__main__":
    main()
