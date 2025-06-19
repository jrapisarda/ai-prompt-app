# 🧠 AI Prompt App

A full-stack, secure, and extensible web application for prompt-based AI interaction, semantic search, and knowledge base augmentation using advanced vector search (ChromaDB) and the OpenAI API.

---

## 🚀 Overview

**AI Prompt App** enables users to interact with GPT-based AI, search previous Q&A sessions semantically, and build a persistent, self-expanding knowledge base.  
The app integrates authentication, prompt logging, semantic retrieval, and PDF document ingestion—serving as a smart assistant for research, Q&A, and document discovery.

---

## ✨ Key Features

- **Prompt-to-AI Q&A**: Securely submit questions to an OpenAI-powered chatbot, with optional web-augmented search.
- **Semantic Search**: Instantly retrieve relevant past conversations using dense vector similarity (ChromaDB, OpenAI Embeddings).
- **Persistent Knowledge Base**: All Q&A pairs are stored and indexed for future discovery.
- **PDF Ingestion**: Bulk-import external documents via PDF for searchable knowledge expansion.
- **User Authentication**: Register, login, and manage access securely (Flask-Login, password hashing).
- **Activity Logging**: All user interactions are logged for traceability and review.
- **Simple, Responsive UI**: Clean interface built with vanilla JS, HTML, and CSS; mobile-friendly and accessible.

---

## 🏗 Architecture

[User]
│
[Web Frontend]
│ ▲
[Flask App / API] <───> [OpenAI API]
│ │
│ ▼
[SQLite DB] [ChromaDB Vector Store]
▲
[PDF Ingestion Pipeline]

markdown
Copy
Edit

- **Flask**: Handles routing, auth, session, and OpenAI API integration.
- **SQLAlchemy**: ORM for user and prompt log storage (SQLite).
- **ChromaDB**: Persistent local vector database for semantic search.
- **OpenAI Embeddings**: Used for both PDF and prompt vectorization.
- **Vanilla JS + HTML/CSS**: Lightweight frontend, easily extendable.

---

## ⚡ Getting Started

### Prerequisites

- Python 3.10+
- [pip](https://pip.pypa.io/)
- [OpenAI API key](https://platform.openai.com/account/api-keys)
- (Optional) [Tiktoken](https://github.com/openai/tiktoken) for PDF ingestion

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/ai-prompt-app.git
cd ai-prompt-app

# Install dependencies
pip install -r requirements.txt

# Copy and configure your environment variables
cp .env.example .env
# Edit .env with your OpenAI API key and model
Running the App
bash
Copy
Edit
# Initialize the database (creates SQLite DB)
python -c "from app import db; db.create_all()"

# Start the Flask app (development mode)
python app.py
App will be available at http://localhost:5000/

📚 PDF Ingestion
Bulk load external PDFs for semantic search:

bash
Copy
Edit
python ingest_pdf.py --pdf your_file.pdf --collection my_collection
Splits and embeds PDF into ChromaDB using OpenAI embeddings.

All data is indexed for seamless retrieval via the app’s semantic search.

🔍 Usage
Register/Login via /register or /login.

Ask questions using the main prompt interface.

Toggle between “AI Query” (OpenAI+Web) and “Semantic Search” (your knowledge base) with one click.

View and select past queries/results in your dashboard.

🛡 Security
Secure password hashing (Werkzeug)

Auth routes protected by Flask-Login

API keys and secrets managed via environment variables

📈 Extensibility & Limitations
Designed for rapid prototyping and research-scale deployments.

Modular: add more endpoints, migrate to Postgres, or deploy on cloud.

No formal test suite or CI/CD (prototype/MVP stage).

Documentation is code-centric; please read code comments and function docstrings for deeper insight.