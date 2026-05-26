"""
Module Name: run_evaluation.py
Description: 基準測試自動化串接進入點 (黃金標準答案完全體)
Author: Ace (Lead Architect)
"""
import asyncio
from dotenv import load_dotenv

# 必須在 Hydra 加載配置前將 .env 倒進 OS Environment
load_dotenv()

import os
import sys
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from src.config.schema import AppConfigSchema
from src.evaluation.pipeline import EvaluationPipeline

async def async_main(validated_config):
    # 🌟 數據平面升級：導入生產級 target_response 黃金標準答案
    mock_eval_dataset = [
        {
            "id": "case_001",
            "prompt": "寫一段 Python 實作二元搜尋演算法。",
            "baseline_response": "def binary_search(arr, x):\n    return arr.index(x)",
            "finetuned_response": "def binary_search(arr: list, target: int) -> int:\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1",
            "target_response": "def binary_search(arr: list, target: int) -> int:\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = low + (high - low) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1"
        },
        {
            "id": "case_002",
            "prompt": "如何預防資料庫 SQL Injection？",
            "baseline_response": "你可以過濾使用者輸入的字串，把單引號取代掉。",
            "finetuned_response": "預防 SQL 注入的最優解是採用參數化查詢（Parameterized Queries）或預編譯陳述式（Prepared Statements）。",
            "target_response": "預防 SQL Injection 的標準防禦方法是全面採用參數化查詢（Parameterized Queries）。所有 SQL 語句應預先編譯（Prepared Statements），確保使用者輸入內容僅被當作數值而非可執行指令。此外，應配合最小權限原則與輸入嚴格驗證。"
        }
    ]

    pipeline = EvaluationPipeline(
        provider_type=validated_config.env.provider_type,
        model=validated_config.hparams.eval_model,
        max_concurrent=validated_config.hparams.max_concurrent_tasks,
        timeout=validated_config.hparams.api_timeout_seconds,
        api_key=validated_config.env.provider_api_key,
        api_url=validated_config.env.eval_api_url
    )

    try:
        results = await pipeline.execute(mock_eval_dataset)
    except Exception as pipe_err:
        print(f"❌ [CRITICAL] Pipeline 執行期嚴重崩潰: {str(pipe_err)}", file=sys.stderr)
        return

    print(f"\n📊 [Pipeline 數據流觀測] 接收到 {len(results)} 筆完工數據。開始建構 Markdown 報告...")

    # 🌟 核心公式重構：分別提取 Baseline 與 Fine-tuned 各自對齊 Target 的平均統計指標
    try:
        avg_base_rouge = float(np.mean([r.get("base_metrics", {}).get("rouge1", 0.0) for r in results]))
        avg_ft_rouge = float(np.mean([r.get("ft_metrics", {}).get("rouge1", 0.0) for r in results]))

        avg_base_bleu = float(np.mean([r.get("base_metrics", {}).get("bleu", 0.0) for r in results]))
        avg_ft_bleu = float(np.mean([r.get("ft_metrics", {}).get("bleu", 0.0) for r in results]))

        avg_base_judge = float(np.mean([r.get("judge_baseline_score", 1) for r in results]))
        avg_ft_judge = float(np.mean([r.get("judge_finetuned_score", 1) for r in results]))
    except Exception as math_err:
        print(f"❌ [指標計算崩潰] 計算平均分時發生錯誤: {str(math_err)}", file=sys.stderr)
        return

    # 安全的字串拼接
    lines = []
    lines.append("# 🏛️ 模組三：LLM 基準測試自動化評估差異報告\n")
    lines.append("## 📊 1. 全局核心效能指標大盤")
    lines.append("| 評估維度 | 原始基準模型 (Baseline) | 微調後優化模型 (Fine-tuned) | 淨提升 / 差異 |")
    lines.append("| :--- | :---: | :---: | :---: |")
    # 🌟 平反大盤：將原本的 "-" 替換為真實指標，並計算出精確的淨提升（Delta）
    lines.append(f"| **平均 ROUGE-1 指標** | {avg_base_rouge:.2f}% | {avg_ft_rouge:.2f}% | **{avg_ft_rouge - avg_base_rouge:+.2f}%** |")
    lines.append(f"| **平均 BLEU 統計分數** | {avg_base_bleu:.2f}% | {avg_ft_bleu:.2f}% | **{avg_ft_bleu - avg_base_bleu:+.2f}%** |")
    lines.append(f"| **LLM-as-a-Judge 平均分** | {avg_base_judge:.1f} / 5.0 | {avg_ft_judge:.1f} / 5.0 | **{avg_ft_judge - avg_base_judge:+.1f}** |\n")
    lines.append("## 🔍 2. 個案盲測詳細追蹤報告\n")

    for r in results:
        lines.append(f"### 📋 案例識別碼: {r.get('id', 'N/A')}")
        lines.append(f"* **使用者提示詞**: `{r.get('prompt', 'N/A')}`")

        b_metrics = r.get('base_metrics', {})
        f_metrics = r.get('ft_metrics', {})
        lines.append(f"* **傳統統計相似度 (對齊標準答案)**:")
        lines.append(f"  * 🔴 Baseline   -> `ROUGE-1: {b_metrics.get('rouge1', 0.0):.1f}%` | `BLEU: {b_metrics.get('bleu', 0.0):.1f}%`")
        lines.append(f"  * 🟢 Fine-tuned -> `ROUGE-1: {f_metrics.get('rouge1', 0.0):.1f}%` | `BLEU: {f_metrics.get('bleu', 0.0):.1f}%`")
        lines.append(f"* **裁判盲測對比**:")

        base_score = r.get('judge_baseline_score', 1)
        base_reason = r.get('judge_baseline_reason', '無有效理由')
        ft_score = r.get('judge_finetuned_score', 1)
        ft_reason = r.get('judge_finetuned_reason', '無有效理由')

        lines.append(f"  * 🔴 **Baseline 評分**: **{base_score} 分**")
        lines.append(f"    * *判分理由*: {base_reason}")
        lines.append(f"  * 🟢 **Fine-tuned 評分**: **{ft_score} 分**")
        lines.append(f"    * *判分理由*: {ft_reason}")
        lines.append("---\n")

    report_content = "\n".join(lines)

    # 實體絕對路徑寫入
    target_path = os.path.join(os.getcwd(), "eval_report_diff.md")
    print(f"💾 正在強行將數據寫入實體磁碟位置: {target_path}")

    try:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(report_content)
            f.flush()
            os.fsync(f.fileno())
        print("\n==================================================")
        print("✨ [Module 03] 基準測試完畢！自動化差異報告已安全落地。")
        print(f"📁 實體絕對路徑: {target_path}")
        print("==================================================\n")
    except Exception as io_err:
        print(f"❌ [IO 寫入崩潰] 無法將檔案寫入磁碟: {str(io_err)}", file=sys.stderr)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    raw_config_dict = OmegaConf.to_container(cfg, resolve=True)
    validated_config = AppConfigSchema.model_validate(raw_config_dict)

    print(f"\n📡 [實時組態] 端點: {validated_config.env.eval_api_url}")
    print(f"🤖 [實時組態] 模型: {validated_config.hparams.eval_model}\n")

    asyncio.run(async_main(validated_config))

if __name__ == "__main__":
    main()