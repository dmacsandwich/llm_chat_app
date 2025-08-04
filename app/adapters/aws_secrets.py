import json
import boto3
from app.core.models import DBSecret

def get_db_secret(secret_id: str, region: str) -> DBSecret:
    client = boto3.client("secretsmanager", region_name=region)
    resp = client.get_secret_value(SecretId=secret_id)  # requires secretsmanager:GetSecretValue
    data = json.loads(resp["SecretString"])
    return DBSecret(
        host=data["host"],
        port=int(data.get("port", 5432)),
        dbname=data["dbname"],
        username=data["username"],
        password=data["password"],
    )
