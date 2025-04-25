# 🧾 Website QA Assistant

Ask questions about any article — like a Medium post — using LangChain, Hugging Face models, and vector search.

This project uses a Retrieval-Augmented Generation (RAG) pipeline to:
- Load content from a public web URL
- Chunk and embed the content
- Store and retrieve context with Chroma
- Generate accurate answers using Hugging Face models (e.g., `flan-alpaca-large`)

---

## 🚀 Features

- 🔗 Load articles using `UnstructuredURLLoader`
- ✂️ Split content with `RecursiveCharacterTextSplitter` for better context retention
- 🧠 Embed using `all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`
- 🗃 Store chunks in a local ChromaDB instance
- 🤖 Answer questions using `flan-alpaca-large` (via `transformers` and `HuggingFacePipeline`)
- 📝 Custom prompt template for high-quality outputs
- 📦 100% local — no OpenAI or hosted API required

---

## 📦 Tech Stack

- [LangChain](https://www.langchain.com/)
- [Transformers (Hugging Face)](https://huggingface.co/docs/transformers)
- [Chroma Vector DB](https://www.trychroma.com/)
- `sentence-transformers`, `tqdm`, `unstructured`, `bs4`

---

## 🛠 Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/your-username/website-qa-assistant.git
cd website-qa-assistant
```
### 2. Create and activate environment

conda create -n data-science-env python=3.10
conda activate data-science-env

### 3. Install dependencies
```bash
pip install -r requirements.txt

```
# 🧪 Usage
## Run the assistant
```bash
python app.py

```
## Example flow
Paste the article URL: https://medium.com/some-article-link
📥 Loading article...
✅ Loaded 1 document(s)
✂️ Splitting document into chunks...
📐 Embedding chunks...
✅ Embeddings stored and retriever ready!
🤖 Loading Hugging Face model...

Ask a question about the article (or type 'exit'): What is the summary?
📖 Answer: This article explains how data science is evolving, not disappearing...

# 📂 Project Structure
.
├── app.py                # Main application script
├── requirements.txt      # Python dependencies
├── .env.example          # Template for environment variable setup
├── README.md             # This file
├── .gitignore            # Ignores .env, chroma_db, etc.
└── chroma_db/            # Auto-generated vector store (excluded from Git)

# 🔐 Environment Variables
Create a .env file in your project root (not committed) with:
HUGGINGFACEHUB_API_TOKEN=your_token_here  # Only needed for hosted models (not used in current setup)
Or just use .env.example as a reference.

# 🧠 Key Concepts
LangChain lets you chain together components like prompts, retrievers, and LLMs

Hugging Face Pipelines give you access to instruction-tuned models like flan-t5 or flan-alpaca

ChromaDB stores embeddings locally and retrieves context at runtime

Custom Prompts ensure the assistant answers accurately and doesn’t hallucinate

# 📜 License
This project is open-sourced under the MIT License.

# 🙋‍♀️ Author
Created by Agnes A