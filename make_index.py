"""將 SQLite 中的新聞摘要編碼成向量並建立 FAISS 索引。"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "news.db"
INDEX_PATH = BASE_DIR / "vector.index"
METADATA_PATH = BASE_DIR / "vector_metadata.json"
DEFAULT_MODEL = "BAAI/bge-m3"


def load_dependencies():
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "缺少向量索引套件。請使用已安裝 FAISS 的專案虛擬環境執行：\n"
            "  .venv\\Scripts\\python.exe cases\\RAGS\\make_index.py"
        ) from exc
    return faiss, np, SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description="由新聞摘要建立 FAISS 向量索引")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence Transformers 模型名稱或本機路徑")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="例如 cpu、cuda；預設由模型套件選擇")
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"找不到資料庫：{DB_PATH}。請先執行 init_db.py。")

    faiss, np, SentenceTransformer = load_dependencies()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, title, summary FROM news "
            "WHERE summary IS NOT NULL AND trim(summary) <> '' ORDER BY id"
        ).fetchall()

    if not rows:
        raise SystemExit("資料庫沒有可建立索引的摘要；請先匯入新聞並產生摘要。")

    ids = np.asarray([row[0] for row in rows], dtype="int64")
    documents = [f"標題：{row[1]}\n摘要：{row[2]}" for row in rows]

    print(f"載入嵌入模型：{args.model}")
    encoder = SentenceTransformer(args.model, device=args.device)
    embeddings = encoder.encode(
        documents,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    # 向量已正規化，內積等同 cosine similarity。每次從資料庫重建，避免重複追加舊向量。
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(embeddings.shape[1]))
    index.add_with_ids(embeddings, ids)

    temp_index = INDEX_PATH.with_suffix(INDEX_PATH.suffix + ".tmp")
    faiss.write_index(index, str(temp_index))
    temp_index.replace(INDEX_PATH)

    metadata = {
        "embedding_model": args.model,
        "metric": "cosine_similarity_via_normalized_inner_product",
        "document_count": len(rows),
        "dimension": int(embeddings.shape[1]),
        "database": DB_PATH.name,
    }
    temp_metadata = METADATA_PATH.with_suffix(METADATA_PATH.suffix + ".tmp")
    temp_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_metadata.replace(METADATA_PATH)

    print(f"索引完成：{len(rows)} 篇新聞，向量維度 {embeddings.shape[1]}")
    print(f"FAISS 索引：{INDEX_PATH}")
    print(f"索引資訊：{METADATA_PATH}")


if __name__ == "__main__":
    main()
