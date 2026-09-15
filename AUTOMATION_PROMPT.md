# Threads 農產品草稿與發布流程

這是 `lin.yusei` 的 Threads 農產品貼文唯一流程規格。排程執行時完整讀取本檔與同目錄 `STYLE_GUIDE.md`，不得依賴聊天中未寫入本檔的舊指示。

## 排程任務

- 每週一至週五台北時間 12:30 執行。
- 同日 13:30、14:30 再檢查交付狀態；當天已完整交付就安靜結束，缺稿或未交付才補跑。排程續查保留在同一個對話。
- 排程只產生草稿並交付到目前 Codex 對話，絕對不自行發布。
- 使用者看過草稿後，才依「草稿修改與發布授權」處理。

## 每日開始與中斷恢復

先執行 `python3 /Users/yuukilin/Documents/Codex/threads-auto-poster/draft_workflow.py --thread-id <目前對話 ID> --record`。
程式只讀本機日報、當日草稿和對話歷史；不連線 Meta、不讀鑰匙圈、不檢查 API 權杖。

- `generate_draft`：今天沒有草稿且找到當日日報，繼續以下完整選題流程。這個預檢不取代下方的完整報告與價格門檻。
- `deliver_saved_draft`：當日檔案存在，但找不到包含該版本全文的已完成最終回覆。直接讀回並交付這一版；不要重新選題、重新查圖或等待使用者核准。
- `skip_delivered`／`skip_published`／`skip_weekend`：不重貼、不重新研究；heartbeat 回覆 DONT_NOTIFY。`awaiting_approval` 不等於未交付；只有同日同版全文在已完成 final 中才算交付成功。
- `await_today_report`：本次說明當日日報尚未就緒，等待下一個續查時段，不使用昨天內容。後續條件完全相同時安靜；14:30 仍缺資料時直接告知缺少項目。
- `repair_draft`：保存既有文字並補齊缺漏或修正無效中繼資料；完成後交付原稿。`review_multiple_drafts` 則依本對話最新版辨認候選稿，避免重複產稿。
- 本機收據位於 `.local/draft-runs/YYYY-MM-DD.json`，記錄當天檔案摘要及實際完成的回覆 ID。不得在送出 final 之前自行填寫「已交付」；由下一次檢查讀取已完成的對話全文核實。
- 若歷史讀取出現索引序號錯誤或只回傳舊訊息，但原始紀錄或檔案已有今日草稿，回報「對話歷史索引異常」，仍在當次 final 貼出完整草稿。不要把它誤診成鑰匙圈或 Meta API 問題。

## 完成時間

以當日日報既有研究為基礎，文字草稿目標在啟動後 15 分鐘內交付。外部來源單站最多重試一次；圖表搜尋最多 3 分鐘，找不到即不附圖。不要因選用的圖表或網站超時，一直延長整個草稿流程。當日數字已在日報有明確日期與來源時，可沿用並在草稿外說明原頁暫時無法重開；核心來源互相矛盾時則換下一個合格事件。

## 當日資料門檻

先以 `Asia/Taipei` 判定日期，完整讀取：

`/Users/yuukilin/.codex/automations/daily-agri-check/last-run.md`

先以 `last-run.md` 為主。日期驗證優先讀 frontmatter 的 `date`；若 `last-run.md` 為了對話交付而省略 frontmatter，則以 H1 標題日期與「資料截至／資料截止」日期共同確認。frontmatter 存在時，三者不得互相衝突。

價格表可接受 `## 今日價格` 或 `## 價格確認`，以表格內的品種、報價日與日變動為準，不綁死單一標題名稱。若 `last-run.md` 缺漏或不完整，再唯讀檢查當日 canonical 日報：

`/Users/yuukilin/Library/Mobile Documents/iCloud~md~obsidian/Documents/卡片筆記盒模板/農產品追蹤/daily-report/YYYY-MM-DD-農產品日報.md`

只有可確認為今天、包含完整讀者報告與可比較價格表時才可出稿。若兩份當日檔案都不存在、不完整、仍在必要官方附件硬暫停或沒有可比較價格，直接說明缺少什麼，不得拿昨天內容冒充今天。

候選品種只限糖、玉米、小麥、大豆、棉花、咖啡、可可與棕櫚油，排除 WTI 原油。價格只用來選題，正文不必寫價格。

## 獨立主題、新鮮度與去重

1. 從完整讀者報告建立獨立主題清單，不綁死 Markdown 標題層級。以下兩種都可候選：`## 市場主線與核心判斷` 下的獨立 `###` 主題；以及日報正文中以品種為中心、具有獨立標題、日期、來源、核心數字與完整因果的 `##` 分析段落，例如 `## 糖：...`、`## 大豆：...`、`## 棕櫚油：...`。
2. 日報寫成「新進展」「持續追蹤」「信心調整」或「未改寫整體主線」，都不直接決定能否出稿。只要今天出現新的原始來源、更新後核心數字或事件結果落地，而且能獨立解釋對單一品種的影響，就算合格新事件。
3. 段落中的附帶比較、沒有新數字的跨品種補充、方向速覽表，以及候選表中的純 `radar`／`watch` 資料，不得單獨升格成 Threads 主題。若一個主題提到多個品種，只有在各品種都有獨立來源與核心數字時才拆分；否則以標題與首段的主要分析對象為歸屬。
4. 讀取 `/Users/yuukilin/Documents/Codex/threads-auto-poster/drafts/` 內全部既有 `.json` 與 `.txt`，並與 Obsidian `農產品追蹤/daily-report/` 內上一個工作日日報比較。
5. 以「品種＋原始來源網址＋來源發布／更新日期＋核心數字」作為事件識別。同一事件或實質相同核心數字曾用於舊草稿，就不得重寫；是否曾發布不影響去重。只有新的原始來源更新、核心數字改變或事件結果落地，才算新進展。
6. 將合格獨立主題所屬品種依可比較報價日的價格絕對變動由大到小排序。優先使用所有候選都有報價的最新共同交易日；缺少共同交易日時，使用各候選最近完整報價並在草稿外說明。第一名若只有舊事件就往下選，不能為了價格第一名回頭使用 tracker 舊聞。
7. 當日日報只要至少有一則未使用的新獨立主題，就必須建立新草稿。只有全部主題都沒有新事件時，才回覆「今日沒有未使用的新事件」，不得重貼舊稿充數。
8. 每個工作日都是獨立批次。昨天或更早的草稿即使仍是 `awaiting_approval`、尚未發送或發布後被刪除，都不能阻擋、取代或延後今天的新草稿。

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

不得讀取、顯示、複製或輸出鑰匙圈權杖。權杖健康檢查與每日草稿是兩條獨立流程：狀態檔不存在、最近一次續期失敗、權杖無效或帳號不符，都只能在草稿外加上不含秘密的警告，絕對不能阻擋、延後或取代當日草稿。若狀態正常且最近成功續期距今未超過 40 天，則不顯示警告。
