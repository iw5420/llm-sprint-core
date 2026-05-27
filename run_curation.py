"""
Module Name: run_curation.py
Description: 模組二：多核心資料去重管線進入點。
             已修正環境變數注入破口，並徹底拔除所有標籤髒資料，確保程式碼綠燈編譯。
Author: Ace (Lead Architect)
"""
import asyncio
import time
from dotenv import load_dotenv

# 🌟 核心防禦：必須在 Hydra 搶佔記憶體前優先將 .env 注入 OS Environment
# 這是修正 PROVIDER_TYPE 找不到的唯一正解
load_dotenv()

import hydra
from omegaconf import DictConfig, OmegaConf
from src.config.schema import AppConfigSchema
from src.data.pipeline import DataCurationPipeline

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # 1. 透過模組一的守門員進行組態與硬體偵測
    # resolve=True 會強制解析 YAML 中的環境變數引用
    raw_config_dict = OmegaConf.to_container(cfg, resolve=True)
    validated_config = AppConfigSchema.model_validate(raw_config_dict)

    print("==================================================")
    print("🚀 [Module 02] 啟動多核心 Data Curation 壓測管線...")
    print(f"📡 專案識別: {validated_config.project_name}")
    print("==================================================")

    # 動態取得模組一感知到的最佳執行緒/進程數與硬體設備名稱
    workers = validated_config.optimal_threads
    device_type = validated_config.device
    perms = validated_config.hparams.num_perm
    bands = validated_config.hparams.num_bands

    # 2. 生成模擬私有文本語料庫
    print("生成模擬私有文本語料庫中...")
    base_text = "This is a production grade high-concurrency system pipeline designed for LLM MLOps training."
    mock_data = []
    for i in range(10000):
        if i % 10 == 0 and i > 0:
            text = f"{base_text} [流水號: {i-1}]"
        else:
            text = f"{base_text} [流水號: {i}]"
        mock_data.append({"id": f"doc_{i}", "text": text})

    # 3. 執行清洗去重管線
    start_time = time.time()
    pipeline = DataCurationPipeline(
        num_workers=workers,
        num_perm=perms,
        num_bands=bands
    )
    clean_data = pipeline.execute(mock_data)
    elapsed = time.time() - start_time

    # 📊 4. 跨平台動態數據計算
    total_raw = len(mock_data)
    total_clean = len(clean_data)
    total_removed = total_raw - total_clean
    mutation_rate = (total_removed / total_raw) * 100 if total_raw > 0 else 0
    throughput = total_raw / elapsed if elapsed > 0 else 0

    # 🛠️ 硬體名稱轉譯
    device_mapping = {
        "mps": "Apple Silicon (MPS 加速驅動)",
        "cuda": "NVIDIA CUDA (高效能 GPU 叢集)",
        "cpu": "Standard Intel/AMD (純 CPU 運算模式)"
    }
    infrastructure_name = device_mapping.get(device_type.lower(), f"Unknown Device ({device_type})")

    # 報告輸出
    width = 55
    print("\n" + "=" * width)
    print(f"{'📊 數據清洗與去重效能報告'.center(width - 6, ' ')}")
    print("=" * width)
    print(f" ⚙️  執行環境配置")
    print(f"    ├── 運算基礎架構 : {infrastructure_name}")
    print(f"    └── 並行工作進程 : {workers} Cores (自動對齊最佳核心數)")
    print(f"    {'-' * (width - 4)}")
    print(f" 📦 數據規模與質量指標")
    print(f"    ├── 原始輸入數據 : {total_raw:,} 筆")
    print(f"    ├── 清洗後留存數 : {total_clean:,} 筆")
    print(f"    ├── 剔除重複數據 : {total_removed:,} 筆")
    print(f"    └── 總體數據去重率: {mutation_rate:.2f}%")
    print(f"    {'-' * (width - 4)}")
    print(f" ⚡ 管線吞吐效能評估")
    print(f"    ├── 管線執行總耗時: {elapsed:.4f} 秒")
    print(f"    └── 峰值計算吞吐率: {throughput:.2f} docs/sec")
    print("=" * width)
    print(f"{'✅ DATA CLEANING & DEDUPLICATION COMPLETE!'.center(width, ' ')}")
    print("=" * width + "\n")

if __name__ == "__main__":
    main()