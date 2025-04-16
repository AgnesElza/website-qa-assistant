from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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