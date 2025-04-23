from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Step 1: Ask for the article URL
url = input("Paste the article URL: ").strip()

# Step 2: Load the article content
print("📥 Loading article...")
loader = UnstructuredURLLoader(urls=[url])
documents = loader.load()
print(f"✅ Loaded {len(documents)} document(s)")

# Step 2.5: Clean the raw text to remove repetitive junk
for doc in documents:
    doc.page_content = doc.page_content.replace(
        "Help\nStatus\nAbout\nCareers\nPress\nBlog\nPrivacy\nRules\nTerms\nText to speech", ""
    )

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
#generator = pipeline("text-generation", model="gpt2", max_new_tokens=100)
#generator = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)
#generator = pipeline("text2text-generation", model="google/flan-t5-large", max_new_tokens=200)
generator = pipeline(
    "text2text-generation", 
    model="declare-lab/flan-alpaca-large", 
    max_new_tokens=200,
    repetition_penalty=1.2, #Avoids repeated phrases
    do_sample=False #Makes output more accurate and consistent
                     )

llm = HuggingFacePipeline(pipeline=generator)

# Step 7: Create QA chain with retriever and LLM
#qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 7: Create a custom prompt to improve answer quality
custom_prompt = PromptTemplate(
    template=(
        "You are an assistant for answering questions about a specific article.\n"
        "Use the following context to answer the question. If the answer isn't in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\nAnswer:"
    ),
    input_variables=["context", "question"]
)

# Step 8: Create QA chain using the custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# Step 9: Ask questions
while True:
    query = input("\nAsk a question about the article (or type 'exit'): ").strip()
    if query.lower() == "exit":
        break
    answer = qa_chain.invoke(query)
    print(f"\n📖 Answer: {answer}")