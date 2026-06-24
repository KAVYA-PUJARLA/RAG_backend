from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware  # 1. Import the middleware
from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os 
class QuestionRequest(BaseModel):
    question: str

load_dotenv()

app = FastAPI()


embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HF_TOKEN"), # Better yet: os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model_name="sentence-transformers/all-MiniLM-L6-v2"
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


# 3. Add the CORS middleware to your FastAPI application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Allows requests from your Angular app
    allow_credentials=True,
    allow_methods=["*"],             # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],             # Allows all headers
)
@app.post("/ask")
def ask_question(request: QuestionRequest):

    results = db.similarity_search(
        request.question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    prompt = f"""
    Answer ONLY from the provided context.

    If answer is not present,
    say 'I don't know'.

    Context:
    {context}

    Question:
    {request.question}
    """

    response = model.invoke(prompt)

    return {
        "answer": response.content,
        "sources": [
            doc.metadata["source"]
            for doc in results
        ]
    }

def main():
    print("Hello from rag-sample!")


if __name__ == "__main__":
    main()
