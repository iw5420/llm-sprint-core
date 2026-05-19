🏛️ LLM Lifecycle Sprint Core (Monorepo)

84小時進修實作核心倉庫：大模型全生命週期（組態 ➔ 清洗 ➔ 訓練 ➔ 對齊 ➔ 評估 ➔ 觀測）端到端工業級管線。

🗺️ 知識庫與實戰日誌對齊 (Obsidian Mapping)

本專案與本地 Obsidian 知識庫之「空間架構線」完全對齊。在 Obsidian 中閱讀時，可透過下方內部連結一鍵跳轉至實戰日誌與開發進度索引：

🧭 實戰日誌導覽進入點： [📂 08_mlops_sprint_logs 實戰日誌首頁](https://github.com/iw5420/ace-obsidian-vault/blob/main/08_mlops_sprint_logs/_sprint_logs_index.md.md)


📝 模組一實戰詳細紀錄： [📄 08.1 模組一：組態與安全驗證日誌](https://github.com/iw5420/ace-obsidian-vault/blob/main/08_mlops_sprint_logs/08.1_mod1_config_%26_safety.md)

🚀 1. 專案核心特色 (Key Features)

本專案採用 Monorepo 單一程式碼庫架構，旨在打破傳統機器學習腳本零散、難以維護的痛點，建立一個具備強型態安全與硬體自適應的自動化 MLOps 系統。

強型態組態守門員： 結合 Hydra 的階層式配置與 Pydantic v2，在進入高耗能訓練前，對 LoRA 等核心參數進行型態與物理邊界校驗。

執行期硬體自我感知： 自動辨識 Apple Silicon (MPS) 或 NVIDIA (CUDA) 加速晶片，並鎖定最佳 CPU 實體核心數，杜絕執行緒過度爭搶。

端到端全生命週期： 涵蓋從最上游的海量數據去重、Packing 打包、手寫 PyTorch 訓練迴圈、DPO 偏好對齊，到最下游的非同步 LLM 盲測與自動化驗證。

📂 2. 專案拓撲結構 (Topology)

llm-sprint-core/
├── config/                  # [模組一] 階層式組態目錄 (YAML)
│   ├── config.yaml          # 全局基礎組態
│   ├── env/                 # 環境覆蓋配置 (dev/prod)
│   └── hparams/             # 微調與 LoRA 超參數配置
├── src/                     # 系統核心源碼基底
│   ├── config/              # [模組一] 組態校驗與硬體偵測
│   ├── data/                # [模組二、六] 數據清洗與流式載入
│   ├── evaluation/          # [模組三] 非同步評估引擎
│   ├── models/              # [模組四、五] 訓練迴圈與損失函數
│   └── utils/               # [模組七] 觀測與 WandB 對接
├── tests/                   # 測試套件
│   ├── unit/                # 各模組獨立單元測試 (pytest)
│   └── test_e2e_lifecycle.py# [模組七] 全生命週期整合測試
├── main.py                  # 系統進入點
└── requirements.txt         # 統一的外部生態系依賴鎖定


⚙️ 3. 快速開始與安裝 (Installation)

3.1 開發環境建置

本專案建議使用 Python 3.10 以上之環境。請在乾淨的虛擬環境中執行以下指令安裝核心依賴：

pip install -r requirements.txt


3.2 執行環境自我檢查 (Sanity Check)

在啟動程序前，可透過以下指令快速驗證本地 PyTorch 加速晶片與系統核心狀態：

python -c "import torch, psutil; print('--- Hardware Status ---'); print('MPS Available:', torch.backends.mps.is_available()); print('Physical Cores:', psutil.cpu_count(logical=False))"


3.3 執行組態核心

執行 main.py 以啟動 Hydra 階層組態解析並通過 Pydantic 的安全防禦閘門：

python main.py


若欲在命令列中動態覆蓋超參數進行實驗，可直接追加引數（無需修改 YAML 實體檔案）：

python main.py hparams.lr=1e-4 hparams.r=16


🗺️ 4. 7大實作模組藍圖 (Roadmap)

本專案隨著 84 小時衝刺進度，依序將以下 7 大核心模組落實於 src/ 中：

模組一：動態組態與型態安全架構 (Done)

實作 Hydra 階層 YAML 覆蓋、Pydantic v2 BaseSettings 超參數邊界防禦與 Apple Silicon 執行期實體執行緒限制。

模組二：多核心加速 Data Curation 管道

使用 Python 多處理器並行洗滌文本，並實作手寫 MinHash + LSH 百萬級海量數據去重。

模組三：非同步自動化基準測試引擎

使用 Python asyncio 實作非同步高併發外部 LLM-as-a-Judge 盲測與自動化 Markdown 差異報告導出。

模組四：手寫 PyTorch + MPS 訓練迴圈

捨棄高階訓練器，自定義客製化 Training Loop，實作梯度累積 (Gradient Accumulation) 與 LoRA 權重 Merge 邏輯。

模組五：DPO 數據管道與損失函數實作

偏好數據結構校驗，純手寫 $L_{DPO}$ 損失函數與矩陣對數機率運算。

模組六：Continued Pre-Training 大文本吞吐

手寫工業級 Packing 打包演算法消除 Padding 浪費，並整合 IterableDataset 實作 Streaming 加載防止記憶體過載。

模組七：工業級 MLOps 觀測與自動化驗證

手寫迴圈原生對接 WandB 指標追蹤，並撰寫一鍵式端到端生命週期整合測試 (test_e2e_lifecycle.py)。
