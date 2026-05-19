🏛️ 專案簡介 (Repository Description)建議填入 GitHub / GitLab 專案設定中的 Description 欄位（控制在 250 字內，精準突顯架構師級別技術棧）：PlaintextProduction-grade LLM Lifecycle Pipeline (Monorepo) featuring Hydra hierarchical configurations, Pydantic v2 type-safe validations, localized hardware-aware thread optimization for Apple Silicon/CUDA, and E2E automation verification.
📄 專案 README.md 原始碼請在你的專案根目錄下建立 README.md，並將以下內容完整貼入：Markdown# 🏛️ LLM Lifecycle Sprint Core (Monorepo)

> 84小時進修實作核心倉：融合 Hydra 階層式組態管理、Pydantic v2 生產級強型態校驗，與多模組端到端（E2E）大模型微調生命周期管線。

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-V2-red.svg)](https://docs.pydantic.dev/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-orange.svg)](https://hydra.cc/)

---

## 🗺️ 知識庫與實戰日誌對齊 (Obsidian Mapping)

本程式碼倉庫（Codebase）與本地 Obsidian 知識庫之「空間架構線」完全對齊。在 Obsidian 中閱讀時，可透過下方連結一鍵跳轉至實戰日誌核心索引：

* **🧭 核心導覽進入點：** `[[_sprint_logs_index.md|📂 08_mlops_sprint_logs 實戰日誌首頁]]`
* **📄 當前模組實戰紀錄：** `[[08.1_mod1_config_&_safety.md|📝 08.1 模組一組態與安全驗證日誌]]`

---

## 🛠️ 1. 專案拓撲結構 (Repository Topology)

本專案採用 **Monorepo** 架構開發，以確保組態、數據清洗、訓練迴圈與自動化驗證之間的強型態閉環與連續性。

```text
llm-sprint-core/             # 專案根目錄
├── config/                  # [模組一] 階層式組態目錄
│   ├── config.yaml          # 全局基礎組態進入點
│   ├── env/
│   │   ├── dev.yaml         # 開發環境覆蓋設定
│   │   └── prod.yaml        # 生產環境覆蓋設定
│   └── hparams/
│       └── ryan_lora.yaml   # LoRA 微調超參數模組
├── src/                     # 核心生產代碼基底
│   ├── __init__.py
│   └── config/              # [模組一] 安全架構與執行期感知
│       ├── __init__.py
│       ├── hardware.py      # Runtime 硬體偵測器 (Apple Silicon/CUDA Thread 鎖定)
│       └── schema.py        # Pydantic v2 強型態驗證守門員
├── main.py                  # 系統執行進入點
└── requirements.txt         # 統一的外部生態系依賴鎖定
🚀 2. 快速開始 (Quick Start)2.1 環境建置與套件安裝建議在乾淨的虛擬環境（如 conda 或 venv）中執行以下指令，一鍵安裝鎖定版本的生產級依賴：Bashpip install -r requirements.txt
requirements.txt 內容參考：pydantic==2.10.0 | hydra-core==1.3.2 | omegaconf==2.3.0 | psutil==5.9.8 | torch==2.3.02.2 執行 Runtime 環境驗證在啟動主程序前，可透過以下單行指令對本地的 PyTorch 加速晶片（MPS/CUDA）與實體核心進行自我檢查（Sanity Check）：Bashpython -c "import torch, psutil; print('--- Runtime Check ---'); print('MPS Available:', torch.backends.mps.is_available()); print('Physical Cores:', psutil.cpu_count(logical=False))"
2.3 啟動組態核心執行 main.py 以啟動 Hydra 階層組態解析並通過 Pydantic 的型態防禦閘門：Bashpython main.py
若欲在命令列（CLI）中動態覆蓋超參數進行實驗，可直接追加引數（無需修改 YAML 實體檔案）：Bashpython main.py hparams.lr=1e-4 hparams.r=16
📈 3. 核心模組實作進度表 (Sprint Roadmap)本倉庫將隨著 84 小時衝刺進度，依序將各核心模組代碼內聚至對應的封裝路徑中：模組一：動態組態與型態安全架構 (Done) ➔ 實作 Hydra + Pydantic v2，落實 Apple Silicon M 系列晶片實體效能核（P-cores）Thread 鎖定調配。模組二：Data Curation 數據清洗管線 (In Progress) ➔ 多核心加速與手寫 MinHash + LSH 百萬級文本去重。模組三：非同步評估引擎 ➔ asyncio 併發呼叫與 LLM-as-a-Judge 盲測。模組四：手寫自訂訓練迴圈 ➔ PyTorch 權重凍結、LoRA 矩陣動態注入與 TensorBoard 觀測。模組五：DPO 偏好對齊 ➔ 純手寫 $L_{DPO}$ 損失函數與矩陣對數機率運算。模組六：Continued Pre-Training 大文本吞吐 ➔ 手寫 Packing 演算法與 IterableDataset 流式加載。模組七：工業級 MLOps 觀測與自動化驗證 ➔ 一鍵 E2E 全生命周期管線回歸測試與 CI/CD-Ready 驗證。🛡️ 4. 邊界破壞測試與防禦表現 (Destructive Testing)本架構在進入耗費算力的訓練迴圈前，會由 src/config/schema.py 進行嚴格的攔截：LoRA Rank 校驗： 若 $r$ 非 2 的冪次方（如設定為 7），系統立即引發 ValidationError 拋出錯誤並中斷程序。學習率邊界校驗： 嚴格限制學習率區間於 [1e-6, 1e-2] 之間，徹底杜絕因筆誤導致初期權重 NaN 的算力浪費。
