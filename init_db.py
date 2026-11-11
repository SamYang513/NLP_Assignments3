"""建立健康新聞資料庫，並將 news_data.json 匯入 SQLite。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "news.db"
DATA_PATH = BASE_DIR / "news_data.json"


def create_database(db_path: Path = DB_PATH) -> None:
    """建立健康新聞資料表；既有資料不會被刪除。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                link TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                article_added_at TEXT,
                reporter TEXT,
                published_date TEXT,
                img_path TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # 若資料庫由舊版複製而來，增補新欄位並沿用原有資料。
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(news)")}
        migrations = {
            "title": "TEXT NOT NULL DEFAULT ''",
            "link": "TEXT NOT NULL DEFAULT ''",
            "content": "TEXT NOT NULL DEFAULT ''",
            "summary": "TEXT NOT NULL DEFAULT ''",
            "article_added_at": "TEXT",
            "reporter": "TEXT",
            "published_date": "TEXT",
            "img_path": "TEXT",
            "created_at": "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
        }
        for column, declaration in migrations.items():
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE news ADD COLUMN {column} {declaration}")
        # 舊資料欄位映射到新版命名，原欄位保留以便相容既有程式。
        if "date" in existing_columns:
            conn.execute("UPDATE news SET published_date = date WHERE published_date IS NULL")
            conn.execute("UPDATE news SET article_added_at = date WHERE article_added_at IS NULL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_added_at ON news(article_added_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_title ON news(title)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_news_link_unique ON news(link)")


def _value(news: dict[str, Any], key: str, default: str = "") -> str:
    value = news.get(key, default)
    return str(value).strip() if value is not None else default


def insert_news_data(data_path: Path = DATA_PATH, db_path: Path = DB_PATH) -> int:
    """匯入新聞；以新聞連結去重，重跑不會產生重複資料。"""
    with data_path.open("r", encoding="utf-8") as file:
        news_data = json.load(file)

    records = []
    for news in news_data:
        link = _value(news, "link")
        title = _value(news, "title")
        if not link or not title or not _value(news, "content"):
            continue
        records.append((
            title,
            link,
            _value(news, "content"),
            _value(news, "summary"),
            # 目前來源 JSON 的 date 是新聞發布日期，不是網站收錄時間，
            # 因此先作為新增時間欄位的可用日期來源，並保留原始發布日期。
            _value(news, "article_added_at") or _value(news, "date") or None,
            _value(news, "reporter") or None,
            _value(news, "published_date") or _value(news, "date") or None,
            _value(news, "img_path") or None,
        ))

    with sqlite3.connect(db_path) as conn:
        conn.executemany("""
            INSERT INTO news (
                title, link, content, summary, article_added_at,
                reporter, published_date, img_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(link) DO UPDATE SET
                title = excluded.title,
                content = excluded.content,
                summary = excluded.summary,
                article_added_at = COALESCE(excluded.article_added_at, news.article_added_at),
                reporter = excluded.reporter,
                published_date = excluded.published_date,
                img_path = excluded.img_path
        """, records)
        return conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]


def main() -> None:
    create_database()
    count = insert_news_data()
    print(f"資料庫已準備完成：{DB_PATH}")
    print(f"目前新聞筆數：{count}")


if __name__ == "__main__":
    main()
