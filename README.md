# 🧠 LangChain + Ollama + FastAPI Backend

This backend service powers an intelligent assistant using:

- **LangChain** for LLM-driven reasoning
- **Ollama** to serve the LLaMA3 model locally
- **FastAPI** as the backend REST API
- **ChromaDB** + HuggingFace embeddings for Retrieval-Augmented Generation (RAG)
- **Custom Tools** (Weather, Developer Info)

---

## 📦 Features

- 🔗 LangChain Agent with multiple tools
- 🌤️ WeatherTool for real-time weather-based suggestions
- 📚 RAG integration using Chroma vector store
- 🚀 FastAPI endpoint to query the agent
- 🧠 Ollama running LLaMA3 locally as LLM
- 📊 Static + real-time data used in agent context

---

## 📁 Folder Structure


---

## ⚙️ Requirements

- Python 3.10+
- Ollama installed with `llama3` model
- ChromaDB
- FastAPI
- Optional: MongoDB (for raw data storage)

Install dependencies:

```bash
pip install -r requirements.txt
