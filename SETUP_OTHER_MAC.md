# 在另一台 Mac 套用

不需要建立第二個 Threads 專案 repo。另一台 Mac 使用同一個 GitHub repo，但 GitHub 登入、Threads 權杖、Codex 排程開關、目標對話與執行紀錄都各自保存在本機。

## 1. 下載專案

在另一台 Mac 的終端機執行：

```bash
mkdir -p ~/Documents/Codex
git clone https://github.com/yuukilin/threads-auto-poster.git ~/Documents/Codex/threads-auto-poster
cd ~/Documents/Codex/threads-auto-poster
```

如果資料夾已存在，改用：

```bash
git -C ~/Documents/Codex/threads-auto-poster pull --ff-only
```

## 2. 安全設定 Threads 權杖與每月續期

```bash
./scripts/setup_mac.sh
```

腳本會跳出 macOS 圖形化安全輸入框，貼上權杖後直接存進鑰匙圈，不會在終端機顯示。接著會唯讀核對帳號 `lin.yusei`、執行測試，並安裝每月 1 日 09:15 的權杖續期工作。權杖不會進入 GitHub、指令參數、草稿或 log。

## 3. 套用 Codex 共用排程定義

先依 `personal-codex-env` 的標準流程完成備份、拉取與安裝：

```bash
cd ~/Documents/Codex/personal-codex-env
./scripts/backup-current.sh
git pull --ff-only
./scripts/install-mac.sh
```

接著在另一台 Mac 的 Codex 開啟 `~/Documents/Codex/threads-auto-poster`，要求 Codex 從：

`~/.codex/automation-templates/threads/automation.toml`

建立或更新「Threads 農產品草稿」排程，並把目標設為當下對話。首次建立先確認本機排程開關；既有排程則保留該 Mac 原本的開關、模型、推理強度、目標與工作目錄。

## 4. 驗證

```bash
cd ~/Documents/Codex/threads-auto-poster
python3 threads_api.py verify
python3 -m unittest discover -s tests -v
launchctl print "gui/$(id -u)/com.yuukilin.threads-token-refresh"
```

看到 `username` 為 `lin.yusei`、測試通過，而且 launchd 工作存在，即完成這台 Mac 的本機設定。不要以測試為由發布貼文。
