# Threads 農產品草稿與發布流程

這是 `lin.yusei` 的 Threads 農產品貼文唯一流程規格。排程執行時完整讀取本檔與同目錄 `STYLE_GUIDE.md`，不得依賴聊天中未寫入本檔的舊指示。

## 排程任務

- 每週一至週五台北時間 12:30 執行。
- 排程只產生草稿並交付到目前 Codex 對話，絕對不自行發布。
- 使用者看過草稿後，才依「草稿修改與發布授權」處理。

## 當日資料門檻

先以 `Asia/Taipei` 判定日期，完整讀取：

`/Users/yuukilin/.codex/automations/daily-agri-check/last-run.md`

只有 frontmatter 的 `date` 等於今天，且包含完整讀者報告與「今日價格」表，才可出稿。若日報不存在、不完整、仍在必要官方附件硬暫停或沒有可比較價格，直接說明缺少什麼，不得拿昨天內容冒充今天。

候選品種只限糖、玉米、小麥、大豆、棉花、咖啡、可可與棕櫚油，排除 WTI 原油。價格只用來選題，正文不必寫價格。

## 獨立主題、新鮮度與去重

1. 先從日報 `## 市場主線與核心判斷` 之下建立獨立主題清單。只有以獨立 `###` 標題完整呈現、且以該品種為中心的主題可以候選。
2. 段落中的附帶比較、跨品種補充句、候選表中的 `radar`／`watch` 資料，不得單獨升格成 Threads 主題。若一個主題提到多個品種，以標題與首段的主要分析對象為歸屬。
3. 讀取 `/Users/yuukilin/Documents/Codex/threads-auto-poster/drafts/` 內全部既有 `.json` 與 `.txt`，並與 Obsidian `農產品追蹤/daily-report/` 內上一個工作日日報比較。
4. 以「品種＋原始來源網址＋來源發布／更新日期＋核心數字」作為事件識別。同一事件或實質相同核心數字曾用於舊草稿，就不得重寫；是否曾發布不影響去重。只有新的原始來源更新、核心數字改變或事件結果落地，才算新進展。
5. 將合格獨立主題所屬品種依當日價格絕對變動由大到小排序。第一名若只有舊事件就往下選，不能為了價格第一名回頭使用 tracker 舊聞。
6. 當日日報只要至少有一則未使用的新獨立主題，就必須建立新草稿。只有全部主題都沒有新事件時，才回覆「今日沒有未使用的新事件」，不得重貼舊稿充數。
7. 每個工作日都是獨立批次。昨天或更早的草稿即使仍是 `awaiting_approval`、尚未發送或發布後被刪除，都不能阻擋、取代或延後今天的新草稿。

## 事件背景

追蹤資料根目錄：

`/Users/yuukilin/Library/Mobile Documents/iCloud~md~obsidian/Documents/卡片筆記盒模板/農產品追蹤`

- 糖：`sugar-tracker.md`、`global-monitor-tracker.md`；氣候直接相關時才讀 `enso-tracker.md`
- 咖啡：`coffee-tracker.md`、`global-monitor-tracker.md`；氣候直接相關時才讀 `enso-tracker.md`
- 可可：`cocoa-tracker.md`、`global-monitor-tracker.md`
- 棕櫚油：`palm-oil-tracker.md`、`global-monitor-tracker.md`；氣候直接相關時才讀 `enso-tracker.md`
- 玉米、小麥、大豆：`crop-progress-tracker.md`、`wasde-tracker.md`、`export-sales-tracker.md`、`drought-monitor-tracker.md`、`plantings-stocks-tracker.md`、`global-monitor-tracker.md`
- 棉花：`cotton-tracker.md`、`crop-progress-tracker.md`、`wasde-tracker.md`、`export-sales-tracker.md`、`drought-monitor-tracker.md`、`global-monitor-tracker.md`

歷史追蹤只補充理解當日事件所需的前值、市場分母、庫存緩衝或直接反證，不得把另一條事件拉進來分散焦點。

## 寫作要求

- 假設讀者沒有看過前文，也不懂這項商品；重點是把事件講清楚，不是逐一解釋名詞。
- 寫出精確發布或事件日期、原先數字與最新數字、變化規模、供需傳導和市場方向。
- 研究底稿可以保守稽核，但 Threads 正文必須根據現有證據選定一個基準方向，直接說明偏多、偏空或影響有限，以及主導力量。
- 禁止以「還要看」「仍待」「尚未公布」「不能排除」「是否會取決於」作結，也不在正文列下一份報告日期、驗證點、資料缺口或研究清單。
- 反證只用來比較力量強弱，例如「乾旱提供支撐，但需求疲弱仍主導」，不能用來迴避結論。
- 明確不等於喊單；不得寫必漲、必跌或虛構確定性。
- 全文沒有最低字數，但連同換行與零寬空格不得超過 Threads 單篇 500 字元。
- 每段承載一個完整意思，通常一至兩句換段。標題與各段之間，在獨立一行放入真正的 `U+200B` 零寬空格；不得把句子切成詞組。
- 價格、字元數、來源、圖片狀態與「尚未發布」放在草稿外。

## 草稿檔案

新稿保存為：

- `drafts/YYYY-MM-DD-commodity.txt`
- `drafts/YYYY-MM-DD-commodity.json`

JSON 至少記錄 `date`、`source_cutoff`、`event_source_url`、`event_published_date`、`event_key`、`commodity`、`daily_change_pct`、`direction`、`revision`、`status=awaiting_approval` 與 `published_post_id=null`。不得覆寫不同日期舊稿；同一天只有使用者要求重跑或修改錯稿時才可覆寫。

## 附圖

附圖可選，而且只接受直接支援核心事件的統計圖、趨勢圖、比較圖或資料表。優先使用原始報告，其次是新聞內嵌圖表；保留來源、標題、期間、單位與必要註解。

不得使用新聞標題截圖、文章文字、商品照片或裝飾圖片，也不得自行重畫圖表。找不到合格圖表時直接寫「本次無合適數據圖表」，不阻擋文字草稿。

## 對話交付

成功出稿時，完整草稿、字元數、圖片與來源狀態、以及「尚未發布」必須直接顯示在使用者可見的對話正文。若本次是 heartbeat，完整內容放在 heartbeat XML 之前；XML 的 message 只放一句短通知。不得把全文塞入通知欄位，也不得只回報完成狀態。

## 草稿修改與發布授權

每次顯示草稿後，該版本成為「目前候選稿」，但不自動發布。

### 使用者貼回修改稿

如果使用者貼回修改後全文、段落或指定修改，而且沒有同時表達發布意圖，就只做最小必要潤飾：保留原意、數字、方向與個人口吻，只修正文句、因果、手機排版及 500 字元限制。把完整潤飾稿再次貼回確認，更新同日 `.txt` 與 JSON `revision`。可來回多次；每次新版本出現後，舊版立即失去發布資格。

### 使用者直接要求發布

如果使用者看過目前候選稿後，明確回覆「發出」「發送」「可以了」「可以發了」「就這版」「幫我發」或其他語意清楚的同義指令，視為對目前候選稿的最終授權。直接發布，不得再問一次。

如果使用者在同一則訊息貼出修改後全文並明確要求發布，只做不改變意思的格式與字數檢查，然後發布該則訊息中的最新版，不必先回傳再等一次確認。

「可以了嗎？」「這樣能發送嗎？」等疑問句、流程討論、引用別人的發布字樣，或無法確定指的是哪一版時，不算授權；只有這類實質歧義才詢問。

發布前把目前候選稿完整寫入對應 `.txt`，確認 JSON 為 `awaiting_approval` 且是本對話最後版本，執行：

`python3 threads_api.py validate --file <草稿檔>`

通過後，把使用者的自然語言授權映射為程式唯一接受的固定安全字串：

`python3 threads_api.py publish --file <草稿檔> --approval 發送`

不得降低程式內的精確核准閘門。發布成功後記錄 post id，把 JSON 更新為 `status=published`、`published_at` 與 `published_post_id`，並回報實際結果。API 失敗或結果不明時不得假稱成功，也不得盲目重發。

已發布內容若再修改，只建立新的修訂草稿；除非使用者再次明確要求，否則不得重發。

## 權杖續期健康檢查

每次排程只讀取：

- `/Users/yuukilin/.codex/automations/threads-token-refresh/last-run.json`
- `/Users/yuukilin/.codex/automations/threads-token-refresh/launchd.log`
- `/Users/yuukilin/.codex/automations/threads-token-refresh/launchd.err`

不得讀取、顯示、複製或輸出鑰匙圈權杖。2026-10-02 以前狀態檔不存在屬正常；之後若最近成功續期距今超過 40 天、帳號不是 `lin.yusei` 或錯誤紀錄顯示失敗，在草稿外加上不含秘密的警告，但不要阻擋出稿。
