import os
import json
import boto3
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


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


def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text


pdf_folder = "data/PDFs"

all_text = ""

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        path = os.path.join(
            pdf_folder,
            file
        )

        print(f"Reading {file}")

        all_text += load_pdf(path)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(all_text)

print(f"Total Chunks: {len(chunks)}")

embedding_model = TitanEmbeddings()

vector_db = FAISS.from_texts(
    chunks,
    embedding_model
)

vector_db.save_local(
    "vectorstore"
)

print("FAISS Vector Store Created Successfully 🚀")