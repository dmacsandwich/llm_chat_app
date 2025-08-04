from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from app.core.models import DBSecret

def make_engine(secret: DBSecret):
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=secret.username,
        password=secret.password,
        host=secret.host,
        port=secret.port,
        database=secret.dbname,
    )
    return create_engine(url, pool_pre_ping=True)
