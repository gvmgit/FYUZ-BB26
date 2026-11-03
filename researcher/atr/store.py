"""Finding-card store. SQLite locally; swap SQLiteStore for a DynamoDB implementation on AWS."""
import json
import sqlite3
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from . import config


class FindingStore(ABC):
    @abstractmethod
    def put(self, card: dict) -> None: ...

    @abstractmethod
    def list(self) -> List[dict]: ...

    @abstractmethod
    def get(self, card_id: str) -> Optional[dict]: ...


class SQLiteStore(FindingStore):
    def __init__(self, path=config.DB_PATH):
        self.path = str(path)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS findings (id TEXT PRIMARY KEY, created REAL, body TEXT)")

    def put(self, card: dict) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO findings VALUES (?, ?, ?)",
                       (card["id"], card.get("created", time.time()), json.dumps(card)))

    def list(self) -> List[dict]:
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT body FROM findings ORDER BY created DESC").fetchall()
        return [json.loads(r[0]) for r in rows]

    def get(self, card_id: str) -> Optional[dict]:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT body FROM findings WHERE id=?", (card_id,)).fetchone()
        return json.loads(row[0]) if row else None
