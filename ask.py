import os
import json
import boto3

from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

load_dotenv()


class TitanEmbeddings(Embeddings):

    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1"
        )

    def embed_documents(self, texts):
        embeddings = []

        for text in texts:
            response = self.client.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({
                    "inputText": text
                })
            )

            result = json.loads(
                response["body"].read()
            )

            embeddings.append(
                result["embedding"]
            )

        return embeddings

    def embed_query(self, text):
        response = self.client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({
                "inputText": text
            })
        )

        result = json.loads(
            response["body"].read()
        )

        return result["embedding"]


embedding_model = TitanEmbeddings()

db = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

question = input("Ask: ")

docs = db.similarity_search(
    question,
    k=3
)

context = "\n\n".join(
    [doc.page_content for doc in docs]
)

prompt = f"""
You are a professional resume assistant.

Answer only from the provided context.
If the answer is not present in the context, say:
'I could not find that information in the document.'

Provide concise and professional answers.

Context:
{context}

Question:
{question}
"""

response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nANSWER:\n")
print(response.choices[0].message.content)