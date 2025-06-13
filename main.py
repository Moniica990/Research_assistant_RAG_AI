import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.document_loaders import TextLoader
import torch


# NEW: Load API key from .env
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Step 1: Initialize ChromaDB
client = chromadb.PersistentClient(path="./research_db")
collection = client.get_or_create_collection(
    name="ml_publications",
    metadata={"hnsw:space": "cosine"}
)

# Step 2: Load publications
def load_research_publications(documents_path):
    documents = []
    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path, file)
            try:
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                print(f"Successfully loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    print(f"\nTotal documents loaded: {len(documents)}")
    publications = []
    for doc in documents:
        publications.append((doc.page_content, doc.metadata["source"]))  # Use source for title if needed
    return publications

# Step 3: Chunk publications
def chunk_research_paper(paper_content, title):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(paper_content)
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "content": chunk,
            "title": title,
            "chunk_id": f"{title}_{i}",
        })
    return chunk_data

# Step 4: Embed documents
def embed_documents(documents: list[str]) -> list[list[float]]:
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )
    embeddings = model.embed_documents(documents)
    return embeddings

# Step 5: Insert into ChromaDB
def insert_publications(collection, publications):
    next_id = collection.count()
    for paper_content, title in publications:
        chunked = chunk_research_paper(paper_content, title)
        chunk_contents = [chunk["content"] for chunk in chunked]
        embeddings = embed_documents(chunk_contents)
        ids = [f"document_{next_id + i}" for i in range(len(chunked))]
        metadatas = [{"title": chunk["title"], "chunk_id": chunk["chunk_id"]} for chunk in chunked]
        collection.add(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
            documents=chunk_contents,
        )
        next_id += len(chunked)

# Step 6: Search the database
def search_research_db(query, collection, embeddings, top_k=5):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )
    query_vector = model.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    relevant_chunks = []
    for i, doc in enumerate(results["documents"][0]):
        relevant_chunks.append({
            "content": doc,
            "title": results["metadatas"][0][i]["title"],
            "similarity": 1 - results["distances"][0][i]
        })
    return relevant_chunks

# Step 7: Generate research-backed answers
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

def answer_research_question(query, collection, embeddings, llm):
    relevant_chunks = search_research_db(query, collection, embeddings, top_k=3)
    context = "\n\n".join([
        f"From {chunk['title']}:\n{chunk['content']}"
        for chunk in relevant_chunks
    ])
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Based on the following research findings, answer the researcher's question:

Research Context:
{context}

Researcher's Question: {question}

Answer: Provide a comprehensive answer based on the research findings above.
"""
    )
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content, relevant_chunks

# Main execution
if __name__ == "__main__":
    # Load and insert publications
    publications = load_research_publications("./documents")
    insert_publications(collection, publications)
    
    # Initialize LLM (requires GROQ API key)
    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)  # Replace with your key
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Example query
    query = "What are effective techniques for handling class imbalance?"
    answer, sources = answer_research_question(query, collection, embeddings, llm)
    
    print("AI Answer:", answer)
    print("\nBased on sources:")
    for source in sources:
        print(f"- {source['title']}")
