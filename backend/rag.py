from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

model = ChatGoogleGenerativeAI(
        model = "gemini-3.5-flash",
        project=os.getenv('GCLOUD_PROJECT_ID')
    )

query = "What is the moon made of?"

results = db.similarity_search(
    query,
    k=3
)
context = "\n\n".join(
    [doc.page_content for doc in results]
)

sources = []

for doc in results:
    sources.append(doc.metadata["source"])

print("Sources:", sources)

prompt = f"""
You are an HR assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
respond with:

I don't know.

Context:
{context}

Question:
{query}
"""

response = model.invoke(prompt)
sample_response = {
   "answer": response.content[0],
   "sources": sources
}
print(sample_response)





