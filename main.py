"""
Module Name: main.py
Description: 模組一：MLOps 強型態組態管理核心。
             已修正橫向組態污染破口，確保全流水線 Bootstrap 順序一致。
Author: Ace (Lead Architect)
"""
import asyncio
import sys
from dotenv import load_dotenv

# 🌟 核心防禦：必須在 Hydra 搶佔記憶體與加載配置前優先執行，橫向平反全域組態引用的環境變數
load_dotenv()

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from src.config.schema import AppConfigSchema

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    print("==================================================")
    print("🚀 [Module 01] 初始化 MLOps 強型態組態管理核心...")
    print("==================================================")

    # 1. 解析組態樹（此時 OS 變數已就位，resolve 順暢通過）
    raw_config_dict = OmegaConf.to_container(cfg, resolve=True)

    try:
        # Pydantic 強型態守門
        validated_config = AppConfigSchema.model_validate(raw_config_dict)
    except Exception as e:
        print("❌ [CRITICAL] 組態校驗未通過！安全阻斷核心程序啟動。", file=sys.stderr)
        print(f"詳細錯誤報告:\n{e}", file=sys.stderr)
        sys.exit(1)

    print("✅ 組態型態安全檢查通過！")
    print(f"|-- 專案名稱: {validated_config.project_name}")
    print(f"|-- 執行硬體: {validated_config.device.upper()}")
    print(f"|-- 計算核心執行緒鎖定: {validated_config.optimal_threads} Threads")
    print(f"|-- LoRA 配置效益: Rank={validated_config.hparams.r}, Alpha={validated_config.hparams.alpha}")
    print(f"|-- 初始學習率: {validated_config.hparams.lr}")
    print("--------------------------------------------------")

    torch.set_num_threads(validated_config.optimal_threads)
    print(f"⚙️  成功將 torch.set_num_threads 鎖定為: {torch.get_num_threads()}")

if __name__ == "__main__":
    main()