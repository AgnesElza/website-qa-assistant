# 🧾 Website QA Assistant

Ask questions about any article using LangChain and Hugging Face models.

This tool lets you input a Medium article URL (or any public webpage), chunk its content, embed it for semantic understanding, and ask questions about it using a free local language model.

## 🚀 Features

- 🌐 Loads web articles directly from a URL
- ✂️ Chunks and stores content in Chroma vector database
- 🧠 Embeds using `all-MiniLM-L6-v2` from Hugging Face
- 🤖 Answers questions using `flan-t5-base` via transformers pipeline
- 🔒 100% local — no OpenAI keys needed
- 💬 Command-line interface (CLI); easy to extend to Streamlit or Gradio

## 📦 Tech Stack

- [`langchain`](https://github.com/langchain-ai/langchain)
- [`transformers`](https://huggingface.co/docs/transformers/)
- [`sentence-transformers`](https://www.sbert.net/)
- [`chromadb`](https://www.trychroma.com/)
- `unstructured`, `beautifulsoup4`, `tqdm`

## 🛠️ Setup

### 1. Clone the repo

git clone git@github.com:AgnesElza/website-qa-assistant.git
cd website-qa-assistant
### 2. Create environment & install dependencies

conda activate your-env-name
pip install -r requirements.txt
### 3. Run the assistant

python app.py
Then paste in a Medium article URL and ask questions about the content!

## 🔮 Example Questions
"What is the main point of the article?"

"Who is the author?"

"List any advice given."

"Summarize the conclusion."

## To-Do / Coming Soon
 Streamlit UI

 PDF and doc support

 Multi-article memory

 Summarization mode

 Export Q&A transcript

## 👩‍💻 Author
Agnes Elza — GitHub Profile

## 📄 License
MIT License
