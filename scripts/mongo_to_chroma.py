import os
from pymongo import MongoClient
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import ssl

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_store")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# Fetch records
print("🚀 Fetching developer data from MongoDB...")
records = list(collection.find({}))
print(f"✅ Total developers fetched: {len(records)}")

# Process records into LangChain Documents
documents = []
for record in records:
    stats = record.get("stats", {})
    text = f"""
    Developer Name: {record.get('name', '')}
    Categories: {', '.join(stats.get('categories', []))}
    Areas: {', '.join(stats.get('service_areas', []))}
    Total Properties: {stats.get('properties_total_count', 0)}
    Created At: {record.get('created_at', '')}
    Slug: {record.get('slug', '')}
    """
    documents.append(
        Document(page_content=text.strip(), metadata={"mongo_id": str(record.get("_id"))})
    )

# Embed with HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create Chroma Vector DB
print("💾 Storing vectors in Chroma...")
Chroma.from_documents(documents, embedding_model, persist_directory=CHROMA_PATH)
print(f"🎉 Successfully stored in: {CHROMA_PATH}")
