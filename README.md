# 透過 RAG 建立健康諮詢小幫手

此專案從健康醫療網新聞分類蒐集文章，使用 SQLite 保存新聞，以文章摘要建立 FAISS 向量索引，最後由本機 Ollama 模型回答問題。網頁介面支援多輪對話，並可逐輪選擇是否搜尋新聞。

## 功能與資料流程

1. 爬取健康醫療網指定新聞分類的列表和文章全文。
2. 將新聞標題、連結、全文、摘要、發布日期等保存到 SQLite。
3. 使用 Ollama 產生摘要（可選）。
4. 使用 `BAAI/bge-m3` 將標題與摘要轉為向量，建立 FAISS 索引。
5. 以向量相似度搜尋新聞，讓 Ollama 根據檢索到的摘要回答。
6. 透過 Flask 網頁進行多輪對話，並顯示本輪檢索到的新聞標題、連結和日期。

資料庫表為 `news`，主要欄位：

| 欄位                 | 說明                       |
| -------------------- | -------------------------- |
| `id`               | 新聞流水號                 |
| `title`            | 新聞標題                   |
| `link`             | 新聞連結，唯一索引避免重複 |
| `content`          | 文章全文                   |
| `summary`          | 文章摘要                   |
| `article_added_at` | 新聞新增時間欄位           |
| `published_date`   | 新聞發布日期               |
| `reporter`         | 記者                       |
| `img_path`         | 新聞圖片網址               |
| `created_at`       | 匯入資料庫的時間           |

目前來源資料只有發布日期，沒有網站收錄時間，因此 `article_added_at` 暫以發布日期填入。回覆會按欄位顯示，不能將此日期視為網站實際上架時間。

## 環境需求

- Windows、Python 3.12（專案 `.venv` 已包含相依套件）
- Ollama 服務
- Ollama 對話模型：預設 `gemma4:26B`；可用 `--model` 選擇本機其他模型
- 嵌入模型：`BAAI/bge-m3`（首次建立索引時需能取得模型）
- FAISS、Sentence Transformers、Flask

在專案根目錄執行以下命令；Windows PowerShell 使用專案虛擬環境 Python：

```powershell
.venv\Scripts\python.exe
```

## 安裝與準備資料

啟動 Ollama，確認所需模型已安裝。若尚未安裝預設聊天模型：

```powershell
ollama pull gemma4:26B
```

初始化或匯入目前的新聞資料：

```powershell
.venv\Scripts\python.exe cases\RAGS\init_db.py
```

爬取新聞列表（先以兩頁檢查分類）：

```powershell
.venv\Scripts\python.exe cases\RAGS\crawl_data.py list --max-pages 2
```

確認 `cases/RAGS/news_list.json` 後抓取文章全文：

```powershell
.venv\Scripts\python.exe cases\RAGS\crawl_data.py articles
```

需要使用本機 Ollama 產生摘要時：

```powershell
.venv\Scripts\python.exe cases\RAGS\crawl_data.py articles --summarize --model gemma4:26B
```

爬蟲會逐筆保存成果，並以新聞連結合併已抓取資料；抓取失敗後可加 `--retry-failed` 重試。預設請求間隔為 2 秒。

## 建立向量索引

摘要存在資料庫後，執行：

```powershell
.venv\Scripts\python.exe cases\RAGS\make_index.py
```

產生：

- `cases/RAGS/vector.index`：FAISS 向量索引
- `cases/RAGS/vector_metadata.json`：嵌入模型、向量維度及文件數

索引使用正規化向量內積進行 cosine similarity 搜尋。重跑會從資料庫重建索引，FAISS ID 對應資料庫 `id`。

## 命令列問答

```powershell
.venv\Scripts\python.exe cases\RAGS\query.py "新聞中提到哪些大腸癌篩檢方式？"
```

指定本機模型：

```powershell
.venv\Scripts\python.exe cases\RAGS\query.py "問題" --model gemma4:26B
```

使用 `--no-retrieval` 可以略過本輪新聞檢索；`--history-json` 可傳入 `role`/`content` 訊息清單作為對話脈絡。

## 啟動網頁問答介面

先確認 Ollama 正在執行，且資料庫、向量索引已準備好。在專案根目錄啟動本機 Flask 服務：

```powershell
.venv\Scripts\python.exe cases\RAGS\app.py
```

使用瀏覽器開啟：[http://127.0.0.1:5000](http://127.0.0.1:5000)

勾選「本輪使用新聞檢索」時，系統會搜尋新聞摘要；取消勾選時，會沿用對話歷史回答但不查詢新聞，並標示本輪未檢索來源。每次回答都會列出本輪檢索到的新聞標題、連結與日期。重新整理頁面會清空目前瀏覽器中的對話。

如需從同一區網的其他裝置存取，可在可信任網路中自行調整 `app.py` 的 host；預設只綁定本機回環位址。

## 目前資料與限制

- 目前資料庫、摘要與向量索引各有 248 篇新聞。
- 新聞資料為健康醫療網內容；此工具整理新聞，不提供個人診斷或治療處方。
- 回答品質取決於新聞涵蓋範圍及本機 Ollama 模型；來源不相關時，模型應說明資料不足。
- 新聞日期目前來自發布日期，不代表網站新增或收錄時間。
