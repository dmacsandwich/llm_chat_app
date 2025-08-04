import json
import boto3
from app.core.ports import EmbedderPort

class TitanEmbedder(EmbedderPort):
    def __init__(self, region: str, model_id: str):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def embed(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text})
        resp = self.client.invoke_model(modelId=self.model_id, body=body)
        payload = json.loads(resp["body"].read())
        return payload["embedding"]

    def embed_batch(self, texts):
        # Simple serial batch; keep it KISS for prototype
        return [self.embed(t) for t in texts]
