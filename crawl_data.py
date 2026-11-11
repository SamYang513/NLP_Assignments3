"""爬取健康醫療網指定新聞分類並整理為可匯入資料庫的 JSON。"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent
SITE_URL = "https://www.healthnews.com.tw"
# 此 UUID 延續原始專案使用的分類網址；可由命令列 --category-url 覆寫。
DEFAULT_CATEGORY_URL = f"{SITE_URL}/channel/9b508eee-a171-75b6-0963-eec4e463fc46"
LIST_PATH = BASE_DIR / "news_list.json"
DATA_PATH = BASE_DIR / "news_data.json"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def read_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"JSON 根節點必須是清單：{path}")
    return data


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temp_path.replace(path)


def parse_date(text: str) -> str | None:
    match = re.search(r"\b(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", text)
    if not match:
        return None
    year, month, day = map(int, match.groups())
    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return None


def scrape_listing_page(
    session: requests.Session, category_url: str, page: int
) -> list[dict[str, Any]]:
    response = session.get(f"{category_url.rstrip('/')}/{page}", timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    items = soup.select("ul.list-unstyled > li.list-item")
    results: list[dict[str, Any]] = []

    for item in items:
        anchor = item.select_one("div.list-title > a[href]") or item.select_one("a[href]")
        if not anchor:
            continue
        info_node = item.select_one("div.list-info")
        info = info_node.get_text(" ", strip=True) if info_node else ""
        title = anchor.get_text(" ", strip=True)
        link = urljoin(SITE_URL, anchor["href"])
        if not title or not link.startswith(SITE_URL):
            continue

        reporter_match = re.search(r"(?:記者|編輯|特派員)\s*[:：]?\s*([^\s|｜]+)", info)
        results.append({
            "link": link,
            "title": title,
            "reporter": reporter_match.group(1) if reporter_match else None,
            "date": parse_date(info),
            "category_url": category_url,
        })
    return results


def crawl_list(category_url: str, max_pages: int | None, delay: float) -> list[dict[str, Any]]:
    session = build_session()
    entries = {item.get("link"): item for item in read_json(LIST_PATH) if item.get("link")}
    page = 1
    consecutive_empty = 0

    while max_pages is None or page <= max_pages:
        try:
            page_items = scrape_listing_page(session, category_url, page)
        except requests.RequestException as exc:
            print(f"第 {page} 頁抓取失敗：{exc}；已保留先前成果。")
            break

        if not page_items:
            consecutive_empty += 1
            print(f"第 {page} 頁沒有符合的新聞。")
            # 有設定頁數時照頁數繼續；自動模式連續空頁便停止。
            if max_pages is None and consecutive_empty >= 2:
                break
        else:
            consecutive_empty = 0
            for item in page_items:
                old = entries.get(item["link"], {})
                # 列表標題或日期較新時更新；文章全文欄位由 crawl_articles 保留。
                entries[item["link"]] = {**old, **item}
            write_json(LIST_PATH, list(entries.values()))
            print(f"第 {page} 頁取得 {len(page_items)} 篇，累計 {len(entries)} 篇。")

        page += 1
        if delay > 0:
            time.sleep(delay)

    result = list(entries.values())
    write_json(LIST_PATH, result)
    return result


def summarize_text(text: str, model: str, ollama_host: str) -> str:
    from ollama import Client

    client = Client(host=ollama_host, timeout=600)
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是健康新聞編輯。根據提供的標題和文章內容，以繁體中文忠實摘要。不得補充文章沒有的醫療建議或事實。",
            },
            {"role": "user", "content": text},
        ],
        options={"temperature": 0.2},
    )
    return response.message.content.strip()


def scrape_article(session: requests.Session, item: dict[str, Any]) -> dict[str, Any]:
    response = session.get(item["link"], timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title_node = soup.select_one("div#article-title > a[href]") or soup.select_one("h1")
    content_node = soup.select_one("div#article-content")
    if not title_node or not content_node:
        raise ValueError("找不到文章標題或全文區塊，網站版面可能已變更。")

    title = title_node.get_text(" ", strip=True)
    content = content_node.get_text("\n", strip=True)
    image_node = soup.select_one("div.container-fluid div.col-12.mb-3 > img[src]")
    image = urljoin(SITE_URL, image_node["src"]) if image_node else None

    # 優先使用文章頁的日期資訊；若找不到則保留列表頁日期。
    date = item.get("date")
    for selector in ("time[datetime]", "meta[property='article:published_time']"):
        node = soup.select_one(selector)
        if node:
            value = node.get("datetime") or node.get("content") or node.get_text(" ", strip=True)
            date = parse_date(value) or date
            if date != item.get("date"):
                break

    return {**item, "title": title, "content": content, "img_path": image, "date": date}


def crawl_articles(
    items: list[dict[str, Any]], *, make_summaries: bool, model: str, ollama_host: str,
    delay: float, retry_failed: bool,
) -> list[dict[str, Any]]:
    existing = {item.get("link"): item for item in read_json(DATA_PATH) if item.get("link")}
    session = build_session()
    total = len(items)

    for index, listing_item in enumerate(items, start=1):
        old = existing.get(listing_item.get("link"), {})
        if old.get("content") and (old.get("summary") or not make_summaries):
            existing[listing_item["link"]] = {**listing_item, **old}
            continue
        if old.get("crawl_error") and not retry_failed:
            continue

        try:
            article = scrape_article(session, {**old, **listing_item})
            article["summary"] = old.get("summary", "")
            article.pop("crawl_error", None)
            if make_summaries:
                article["summary"] = summarize_text(
                    f"標題：{article['title']}\n\n全文：{article['content']}", model, ollama_host
                )
            existing[article["link"]] = article
            print(f"[{index}/{total}] 完成：{article['title']}")
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            existing[listing_item["link"]] = {
                **old, **listing_item, "crawl_error": str(exc),
            }
            print(f"[{index}/{total}] 失敗：{listing_item['link']}：{exc}")
        finally:
            write_json(DATA_PATH, list(existing.values()))

        if delay > 0:
            time.sleep(delay)

    result = list(existing.values())
    write_json(DATA_PATH, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="爬取健康醫療網新聞分類")
    parser.add_argument("stage", choices=("list", "articles", "all"), nargs="?", default="all")
    parser.add_argument("--category-url", default=DEFAULT_CATEGORY_URL, help="新聞分類網址")
    parser.add_argument("--max-pages", type=int, default=None, help="最多抓取頁數；預設自動抓到連續兩頁無資料")
    parser.add_argument("--delay", type=float, default=2.0, help="每次請求間隔秒數")
    parser.add_argument("--summarize", action="store_true", help="使用 Ollama 產生摘要")
    parser.add_argument("--model", default="gemma4:e4b", help="Ollama 模型名稱")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama API 位址")
    parser.add_argument("--retry-failed", action="store_true", help="重試先前失敗的文章")
    args = parser.parse_args()

    if args.stage in ("list", "all"):
        items = crawl_list(args.category_url, args.max_pages, args.delay)
    else:
        items = read_json(LIST_PATH)
        if not items:
            parser.error("news_list.json 沒有新聞，請先執行 list 階段。")

    if args.stage in ("articles", "all"):
        articles = crawl_articles(
            items,
            make_summaries=args.summarize,
            model=args.model,
            ollama_host=args.ollama_host,
            delay=args.delay,
            retry_failed=args.retry_failed,
        )
        print(f"文章資料已寫入 {DATA_PATH}，共 {len(articles)} 篇。")


if __name__ == "__main__":
    main()
