# ================= main.py ================= 
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from core.config import embeddings, llm


PDF_FILE_PATH = "D:/02-Projects/Completed Projects/PDF-RAG-Chatbot/data/attention-is-all-you-need.pdf"
VECTOR_DB_DIRECTORY = "vector_db"

pdf_loader = PyPDFLoader(PDF_FILE_PATH)
documents = pdf_loader.load()

print(f"{len(documents)} pages loaded.")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_chunks = text_splitter.split_documents(documents)

print(f"{len(splitted_chunks)} chunks created.")

chroma_db_vector_store = Chroma.from_documents(
    documents = splitted_chunks,
    embedding=embeddings,
    persist_directory=VECTOR_DB_DIRECTORY
)

retriever = chroma_db_vector_store.as_retriever(
    search_type = "similarity",
    k = 2
)

qa_chain = RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

while True:
    print("-"*50)
    query = input("Enter your query ('0' to exit): ")
    print("-"*50)
    print()
    if query == "0":
        break
    
    llm_response = qa_chain.invoke(query)
    print(llm_response['result'])
    print()