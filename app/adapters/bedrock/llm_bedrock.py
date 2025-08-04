import boto3
from app.core.ports import ChatLLMPort

class LlamaChatLLM(ChatLLMPort):
    def __init__(self, region: str, model_id: str, temperature: float = 0.2, max_tokens: int = 512):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict]) -> str:
        system_blocks = [c for m in messages if m.get("role")=="system" for c in m.get("content",[])]
        ua_messages = [m for m in messages if m.get("role") in ("user","assistant")]
        if not ua_messages or ua_messages[-1]["role"] != "user":
            raise ValueError("Converse requires the last message to be role='user'.")

        resp = self.client.converse(
            modelId=self.model_id,
            messages=ua_messages,
            system=system_blocks or None,
            inferenceConfig={"temperature": self.temperature, "maxTokens": self.max_tokens},
        )
        return "".join(b.get("text","") for b in resp["output"]["message"]["content"])
