"""RAG 問答核心：以新聞摘要檢索來源，交由本機 Ollama 生成回答。"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "news.db"
INDEX_PATH = BASE_DIR / "vector.index"
METADATA_PATH = BASE_DIR / "vector_metadata.json"
DEFAULT_MODEL = "gemma4:26B"


@dataclass
class Source:
    id: int
    title: str
    link: str
    added_at: str | None
    summary: str
    score: float


class RAGAssistant:
    def __init__(
        self,
        *,
        llm_model: str = DEFAULT_MODEL,
        ollama_host: str = "http://localhost:11434",
        embedding_model: str | None = None,
        top_k: int = 5,
    ) -> None:
        self.llm_model = llm_model
        self.top_k = top_k
        self.db_path = DB_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(f"找不到資料庫：{self.db_path}")
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"找不到向量索引：{INDEX_PATH}。請先執行 make_index.py。")
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"找不到索引資訊：{METADATA_PATH}。請重新建立索引。")

        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.embedding_model_name = embedding_model or metadata["embedding_model"]

        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            from ollama import Client
        except ImportError as exc:
            raise RuntimeError(
                "缺少 RAG 執行套件，請使用專案 .venv 執行。"
            ) from exc

        self.faiss = faiss
        self.encoder = SentenceTransformer(self.embedding_model_name)
        self.index = faiss.read_index(str(INDEX_PATH))
        if self.index.ntotal == 0:
            raise ValueError("向量索引目前是空的，請先建立索引。")
        self.client = Client(host=ollama_host, timeout=600)

    def retrieve(self, question: str, top_k: int | None = None) -> list[Source]:
        question = question.strip()
        if not question:
            return []
        query_vector = self.encoder.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        )
        scores, ids = self.index.search(query_vector, top_k or self.top_k)
        found: list[Source] = []
        with sqlite3.connect(self.db_path) as conn:
            for score, news_id in zip(scores[0], ids[0]):
                if news_id < 0:
                    continue
                row = conn.execute(
                    """SELECT id, title, link,
                              COALESCE(article_added_at, published_date, date), summary
                       FROM news WHERE id = ?""",
                    (int(news_id),),
                ).fetchone()
                if row:
                    found.append(Source(
                        id=row[0], title=row[1], link=row[2],
                        added_at=row[3], summary=row[4], score=float(score),
                    ))
        return found

    @staticmethod
    def _history_messages(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
        if not history:
            return []
        messages = []
        for item in history[-12:]:
            role = item.get("role")
            content = item.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        return messages

    def answer(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        use_retrieval: bool = True,
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("請輸入問題。")

        sources = self.retrieve(question) if use_retrieval else []
        if use_retrieval and not sources:
            evidence = "沒有找到可用的新聞摘要。請說明目前資料中沒有足夠來源可回答。"
        elif sources:
            evidence = "\n\n".join(
                f"[來源 {i}] 標題：{src.title}\n摘要：{src.summary}"
                for i, src in enumerate(sources, start=1)
            )
        else:
            evidence = "本輪未檢索新聞資料。若問題涉及本地資料庫中的特定新聞，請明確說明目前沒有查證來源。"

        system_prompt = (
            "你是以繁體中文回答的健康新聞整理助理。你的回答只根據提供的新聞摘要與對話脈絡。"
            "不得把新聞內容延伸成個人診斷、治療處方或未經來源支持的醫療事實。"
            "若來源不足，請清楚說明，不要猜測。來源清單會由系統另外附上。"
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._history_messages(history))
        messages.append({
            "role": "user",
            "content": f"新聞依據如下：\n{evidence}\n\n使用者問題：{question}",
        })

        response = self.client.chat(
            model=self.llm_model,
            messages=messages,
            options={"temperature": 0.3},
        )
        return {
            "answer": response.message.content.strip(),
            "sources": [asdict(source) for source in sources],
            "used_retrieval": use_retrieval,
        }


def format_response(result: dict[str, Any]) -> str:
    lines = [result["answer"], "", "參考新聞："]
    if not result["sources"]:
        lines.append("- 本輪未檢索新聞來源。")
    else:
        for source in result["sources"]:
            lines.extend([
                f"- 新聞標題：{source['title']}",
                f"  新聞連結：{source['link']}",
                f"  文章新增時間：{source['added_at'] or '資料未提供'}",
            ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="健康新聞 RAG 問答")
    parser.add_argument("question", nargs="?", help="要詢問的健康新聞問題")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama 對話模型")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-retrieval", action="store_true", help="本輪不檢索新聞")
    parser.add_argument("--history-json", help="先前對話 JSON 檔，格式為 role/content 清單")
    args = parser.parse_args()

    question = args.question or input("請輸入問題：").strip()
    history = json.loads(Path(args.history_json).read_text(encoding="utf-8")) if args.history_json else []
    assistant = RAGAssistant(llm_model=args.model, ollama_host=args.ollama_host, top_k=args.top_k)
    result = assistant.answer(question, history=history, use_retrieval=not args.no_retrieval)
    print(format_response(result))


if __name__ == "__main__":
    main()
