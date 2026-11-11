"""Flask 網頁版健康新聞 RAG 對話介面。"""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from query import DEFAULT_MODEL, RAGAssistant


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
_assistant: RAGAssistant | None = None


def get_assistant() -> RAGAssistant:
    global _assistant
    if _assistant is None:
        _assistant = RAGAssistant()
    return _assistant


@app.get("/")
def home():
    return render_template("index.html", model_name=DEFAULT_MODEL)


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "請先輸入問題。"}), 400
    if len(question) > 4000:
        return jsonify({"error": "問題太長，請控制在 4000 字以內。"}), 400

    history = payload.get("history", [])
    if not isinstance(history, list):
        return jsonify({"error": "對話歷史格式不正確。"}), 400
    use_retrieval = payload.get("use_retrieval", True) is True

    try:
        result = get_assistant().answer(
            question,
            history=history,
            use_retrieval=use_retrieval,
        )
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("RAG 回答失敗")
        return jsonify({"error": f"問答處理失敗：{exc}"}), 503


if __name__ == "__main__":
    # 僅供本機使用；正式部署時再設定受信任的 WSGI 伺服器。
    app.run(host="127.0.0.1", port=5000, debug=False)
