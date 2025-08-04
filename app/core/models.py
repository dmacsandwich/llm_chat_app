from dataclasses import dataclass

@dataclass(frozen=True)
class DBSecret:
    host: str
    port: int
    dbname: str
    username: str
    password: str
