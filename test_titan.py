import boto3
import json

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

text = "Oji loves building AI applications"

response = client.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=json.dumps({
        "inputText": text
    })
)

result = json.loads(response["body"].read())

print("Embedding length:", len(result["embedding"]))