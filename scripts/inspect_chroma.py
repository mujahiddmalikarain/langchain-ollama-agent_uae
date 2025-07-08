from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
import os

# Load .env variables (if any)
load_dotenv()

# Path to your Chroma vector DB
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")

# Embedding function (can switch to HuggingFaceEmbeddings if preferred)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load Chroma vector store
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Search query example
query = "Top real estate developers in Dubai Marina"
print(f"🔍 Searching for: {query}\n")

# Run similarity search
results = db.similarity_search(query, k=5)

# Show results
print("🧠 Top matching results:\n")
for i, result in enumerate(results):
    print(f"{i + 1}. {result.page_content}\n")
