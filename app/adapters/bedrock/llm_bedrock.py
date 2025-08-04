import boto3
from app.core.ports import ChatLLMPort

class LlamaChatLLM(ChatLLMPort):
    def __init__(self, region: str, model_id: str, temperature: float = 0.2, max_tokens: int = 512):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict]) -> str:
        # messages: [{"role": "system"|"user"|"assistant", "content": [{"text": "..."}]}]
        # Per Bedrock Converse API
        resp = self.client.converse(
            modelId=self.model_id,
            messages=messages,
            inferenceConfig={"temperature": self.temperature, "maxTokens": self.max_tokens}
        )
        # Concatenate returned text blocks
        parts = []
        for block in resp["output"]["message"]["content"]:
            if "text" in block:
                parts.append(block["text"])
        return "".join(parts)
