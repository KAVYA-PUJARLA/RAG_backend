from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

query = "How do I claim insurance?"

results = db.similarity_search(
    query,
    k=3
)

for i, doc in enumerate(results, start=1):

    print("\n")
    print("="*50)

    print(f"Result {i}")

    print(doc.page_content)

    print(doc.metadata)