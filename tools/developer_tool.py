from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

# Load embedding + Chroma DB
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_db = Chroma(
    persist_directory="./chroma_db_store",
    embedding_function=embedding
)

# RAG Prompt
prompt_template = PromptTemplate.from_template("""
You are an expert assistant for real estate developer information.

Context:
{context}

Question:
{name}

Answer:
""")

llm = Ollama(model="llama3")
rag_chain = LLMChain(llm=llm, prompt=prompt_template)

def get_developer_info(name: str) -> str:
    docs = chroma_db.similarity_search(name, k=2)
    if not docs:
        return f"No matching developer info found for: {name}"
    context = "\n\n".join(doc.page_content for doc in docs)
    return rag_chain.run({"context": context, "name": name})
