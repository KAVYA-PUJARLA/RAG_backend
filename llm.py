from langchain.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(
        model = "gemini-3.5-flash",
        project=os.getenv('GCLOUD_PROJECT_ID')
    )

response = model.invoke(
    "What is Artificial Intelligence?"
)

print(response.content)  
