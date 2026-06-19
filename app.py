import streamlit as st
import os
import json
import boto3

from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
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

@st.cache_resource
def load_db():
    return FAISS.load_local(
        "vectorstore",
        embedding_model,
        allow_dangerous_deserialization=True
    )

db = load_db()
st.set_page_config(
    page_title="AWS Bedrock File Reviewer",
    page_icon="📄",
    layout="wide"
)
if "messages" not in st.session_state:
    st.session_state.messages = []
st.title("📄 AWS Bedrock File Reviewer")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
question = st.chat_input(
    "Ask about the uploaded document..."
)
with st.sidebar:

    st.title("📄 File Reviewer")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:

        os.makedirs("data/PDFs", exist_ok=True)

        # Remove old PDFs
        for f in os.listdir("data/PDFs"):
            os.remove(os.path.join("data/PDFs", f))

        file_path = os.path.join(
            "data/PDFs",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(
            f"{uploaded_file.name} uploaded successfully!"
        )
if st.button("Process PDF"):

    os.system("python ingest.py")

    st.success("Vector database created successfully!")

    load_db.clear()
    st.rerun()
if question:

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.spinner("Analyzing document..."):

        docs = db.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are a professional document assistant.

Answer only from the provided context.

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

        answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.expander("Retrieved Context"):
        st.write(context)

    