from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

documents=[]
#converting all documents into langchain documents
for filename in os.listdir("data"):
    reader = PdfReader(f"data/{filename}")
    docs=[]
    for i,page in enumerate(reader.pages):
        text=page.extract_text()
        docs.append(Document(page_content=text,metadata={"page":i+1,"source":f"{filename}"}))
        documents.extend(docs)

#converting document text to chunks
splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks= splitter.split_documents(documents)

#chunks to Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vector = embeddings.embed_query(
    "How many leaves are allowed?"
)
#storing chunks to vector db(faiss)
db = FAISS.from_documents(
    chunks,
    embeddings
)
db.save_local("vectorstore")