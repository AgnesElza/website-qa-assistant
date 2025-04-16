from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Step 1: Ask for the article URL
url = input("Paste the Medium article URL: ").strip()

# Step 2: Load the article content
print("📥 Loading article...")
loader = UnstructuredURLLoader(urls=[url])
documents = loader.load()
print(f"✅ Loaded {len(documents)} document(s)")

# Step 3: Split the text into chunks
print("✂️ Splitting document into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"✅ Split into {len(chunks)} chunks")

# Step 4: Initialize Hugging Face embeddings
print("📐 Embedding chunks...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #small, fast, and accurate

# Step 5: Store in Chroma vector DB
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db")
retriever = db.as_retriever()
print("✅ Embeddings stored and retriever ready!")

# Step 6: Set up the text generation model
print("🤖 Loading Hugging Face model...")
generator = pipeline("text-generation", model="gpt2", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=generator)

# Step 7: Create QA chain with retriever and LLM
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 8: Ask questions
while True:
    query = input("\nAsk a question about the article (or type 'exit'): ").strip()
    if query.lower() == "exit":
        break
    answer = qa_chain.run(query)
    print(f"\n📖 Answer: {answer}")