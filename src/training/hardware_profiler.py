import os
import sys
import torch
import psutil

class RuntimeHardwareProfiler:
    """全平台自適應執行期資源觀測器"""

    @staticmethod
    def log_memory_snapshot(device: str):
        """
        根據動態感知到的 device 類型，自適應抓取正確的記憶體指標，杜絕無效 API 調用。
        """
        device_lower = device.lower()

        # 分支 A：Apple Silicon MPS 架構
        if "mps" in device_lower and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            allocated_bytes = torch.mps.current_allocated_memory()
            print(f"🧠 [Unified Memory Profiler] 當前 Apple MPS 記憶體佔用: {allocated_bytes / (1024 * 1024):.2f} MB")

        # 分支 B：NVIDIA CUDA 雲端 GPU 架構
        elif "cuda" in device_lower and torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated_bytes = torch.cuda.memory_allocated()
            print(f"⚡ [VRAM Profiler] 當前 NVIDIA CUDA 顯示記憶體佔用: {allocated_bytes / (1024 * 1024):.2f} MB")

        # 分支 C：純 CPU 邊緣伺服器環境
        else:
            process = psutil.Process(os.getpid())
            print(f"💻 [System Memory Profiler] 當前 Host CPU 實體記憶體佔用: {process.memory_info().rss / (1024 * 1024):.2f} MB")

        sys.stdout.flush()