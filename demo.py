"""執行兩題健康新聞 RAG 示範，並彙整問題重點與回答。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from query import DEFAULT_MODEL, RAGAssistant, format_response


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "demo_results.md"
DEMO_QUESTIONS = [
    "新聞中提到哪些大腸癌篩檢方式，以及哪些族群應留意篩檢？",
    "新聞中提到腦瘤可能有哪些症狀，出現哪些情況時應就醫檢查？",
]


def summarize_demo(
    assistant: RAGAssistant,
    questions: list[str],
    results: list[dict[str, Any]],
) -> str:
    source_context = []
    for question, result in zip(questions, results):
        sources = "\n".join(
            f"- {source['title']}（{source['link']}；日期：{source['added_at'] or '資料未提供'}）"
            for source in result["sources"]
        ) or "- 沒有檢索到來源"
        source_context.append(
            f"問題：{question}\n回答：{result['answer']}\n檢索來源：\n{sources}"
        )

    response = assistant.client.chat(
        model=assistant.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是健康新聞整理助理。請將兩個問題與其回答整合成繁體中文重點摘要。"
                    "先用條列整理兩個問題各自關心的重點，再作整體回答。只可根據提供內容，"
                    "來源不支持的結論要指出，不能提供個人診斷或治療處方。"
                ),
            },
            {"role": "user", "content": "\n\n".join(source_context)},
        ],
        options={"temperature": 0.2},
    )
    return response.message.content.strip()


def build_report(
    questions: list[str], results: list[dict[str, Any]], summary: str
) -> str:
    sections = [
        "# 健康新聞 RAG 問答示範",
        "",
        f"產生時間：{datetime.now().astimezone().isoformat(timespec='minutes')}",
        "",
        "## 問題重點彙整與整體回答",
        "",
        summary,
    ]
    for index, (question, result) in enumerate(zip(questions, results), start=1):
        sections.extend(["", f"## 問題 {index}：{question}", "", result["answer"], "", "### 檢索新聞來源", ""])
        if not result["sources"]:
            sections.append("- 本題沒有檢索到新聞來源。")
            continue
        for source in result["sources"]:
            sections.extend([
                f"- 新聞標題：{source['title']}",
                f"  新聞連結：{source['link']}",
                f"  文章新增時間：{source['added_at'] or '資料未提供'}",
            ])
    return "\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="兩題健康新聞 RAG 問答與彙整示範")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama 對話模型")
    parser.add_argument("--top-k", type=int, default=3, help="每題檢索新聞數")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--question", action="append", dest="questions", help="自訂問題，可重複兩次")
    args = parser.parse_args()

    questions = args.questions or DEMO_QUESTIONS
    if len(questions) != 2:
        parser.error("請提供剛好兩個 --question，或不指定以使用內建示範問題。")

    assistant = RAGAssistant(llm_model=args.model, ollama_host=args.ollama_host, top_k=args.top_k)
    results = []
    for index, question in enumerate(questions, start=1):
        print(f"\n問題 {index}：{question}")
        result = assistant.answer(question, use_retrieval=True)
        results.append(result)
        print(format_response(result))

    summary = summarize_demo(assistant, questions, results)
    print("\n兩題重點彙整與整體回答：\n")
    print(summary)

    report = build_report(questions, results, summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\n示範紀錄已儲存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
