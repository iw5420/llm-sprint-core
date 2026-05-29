"""
Module Name: run_training.py
Description: 模組四跨平台通用訓練管線拉起進入點（職責清晰，零裸奔完全體）
Author: Ace (Lead Architect)
"""
import asyncio
from dotenv import load_dotenv

# SSOT Bootstrap 鐵律防禦：最頂層強制拉高環境變數加載時間差位置
load_dotenv()

import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from src.config.schema import AppConfigSchema
from src.training.lora_model import MockLinearWithLoRA
from src.training.pipeline import CustomTrainingLoopPipeline

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    print("==================================================")
    print("🚀 [Module 04] 啟動 MLOps 生產級強型態訓練迴路...")
    print("==================================================")

    # ----------------------------------------------------------------------
    # 步驟一：組態反序列化與硬體安全鎖定
    # ----------------------------------------------------------------------
    raw_config_dict = OmegaConf.to_container(cfg, resolve=True)
    validated_config = AppConfigSchema.model_validate(raw_config_dict)

    device = validated_config.device
    threads = validated_config.optimal_threads
    hparams = validated_config.hparams

    torch.set_num_threads(threads)
    print(f"⚙️  硬體系統鎖定：裝置={device.upper()} | 執行緒={torch.get_num_threads()} Threads")

    # ----------------------------------------------------------------------
    # 步驟二：生成模擬特徵資料集 (Data Plane)
    # ----------------------------------------------------------------------
    print("🧬 正在記憶體中生成 2,000 筆高維特徵數據...")
    X_train = torch.randn(2000, 128)
    Y_train = torch.randn(2000, 64)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)

    # ----------------------------------------------------------------------
    # 步驟三：建構封裝模型物件與傳送帶控制管線
    # ----------------------------------------------------------------------
    model = MockLinearWithLoRA(in_features=128, out_features=64, r=hparams.r, alpha=hparams.alpha)
    pipeline = CustomTrainingLoopPipeline(
        model=model,
        lr=hparams.lr,
        grad_accum_steps=hparams.gradient_accumulation_steps,
        max_grad_norm=hparams.max_grad_norm,
        device=device
    )

    # ----------------------------------------------------------------------
    # 步驟四：多型自適應配置 Telemetry 觀測上下文，100% 抹平版本 Drift 地雷
    # ----------------------------------------------------------------------
    if device == "mps":
        print("🍏 偵測到 Mac 晶片！正式啟動 Apple 原生 Metal 核心觀測器")
        profiler_ctx = torch.mps.profiler.profile(mode="interval", wait_until_completed=True)
    elif device == "cuda":
        print("⚡ 偵測到 NVIDIA 晶片！啟動標準 Kineto CUDA 觀測器")
        profiler_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/train_trace'),
            record_shapes=True, profile_memory=True, with_stack=True
        )
    else:
        print("💻 純 CPU 環境，啟用標準通用觀測器")
        profiler_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        )

    # ----------------------------------------------------------------------
    # 步驟五：正式驅動手寫訓練迴圈管線
    # ----------------------------------------------------------------------
    with profiler_ctx:
        for epoch in range(1, hparams.epochs + 1):
            avg_loss = pipeline.train_epoch(train_dataloader, epoch_idx=epoch)
            print(f"✨ [Epoch 完工總結] 輪次: {epoch}/{hparams.epochs} -> 平均綜合收斂 Loss: {avg_loss:.6f}")

            # Kineto Profiler 需要手動踩踏前進，原生 MPS 不需要
            if device != "mps" and hasattr(profiler_ctx, "step"):
                profiler_ctx.step()

    print("✅ [Profiler] 全平台適應型算力 Trace 完工落地。")

    # ----------------------------------------------------------------------
    # 步驟六：一鍵指令 ➔ 模型，請將你的活跃增量結晶檔案落地
    # ----------------------------------------------------------------------
    model.save_lora_weights(folder_path=hparams.save_dir, filename="ryan_lora_checkpoint.pt")

    # ----------------------------------------------------------------------
    # 步驟七：一鍵指令 ➔ 模型，請執行內部熔斷合併，永久回收線上推理效能
    # ----------------------------------------------------------------------
    print("⚔️  觸發終極微調完工合併指令：model.merge_and_unload()...")
    model.merge_and_unload()

    print("🎉 模組四全鏈路完全體整合完工！主流程實現高階去耦合閉環。")
    print("==================================================\n")

if __name__ == "__main__":
    main()