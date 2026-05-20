import os
import psutil
import torch

class RuntimeHardwareDetector:
    """
    Runtime 硬體感知調度器
    負責自動偵測加速晶片類型，並精確計算核心並行度，防止多核心環境下的 Context Switch 性能過載。
    """
    
    @staticmethod
    def detect_device() -> str:
        """偵測當前最優的加速硬體運算裝置"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def calculate_optimal_threads(device: str) -> int:
        """
        根據運行裝置（加速晶片）與作業系統拓撲，動態計算最優 CPU Thread 數量。
        
        頂級企業通用最佳化邏輯與效能防線：
        1. MPS (Apple Silicon - 以當前 M4 Pro 為例):
           - 效能陷阱：若直接呼叫 Python 內建的 `os.cpu_count()`，系統會回報「邏輯核心數」（在 M 系列晶片上
             包含 P-cores 效能核心與 E-cores 節能核心，例如 M4 Pro 會直接回報 12 或更高）。
           - 負面開銷：深度學習的密集矩陣運算（如 PyTorch 的張量計算）若分配到 E-cores，非但無法加速，
             反而會因為「木桶效應」拖慢整體運算時脈，並引發極高頻的 Context Switch（上下文切換）損耗。
           - 最佳化解法：必須透過 `psutil.cpu_count(logical=False)` 強制過濾 E-cores，精確取用真實的
             「物理核心數」（Physical Cores），並將 `torch.set_num_threads` 緊緊錨定於此，最大化每瓦效能表現。
             
        2. CUDA (NVIDIA):
           - 最佳化解法：矩陣計算已全數移交 GPU，此時 CPU 主要負責 Dataloader 的資料預處理與 I/O 吞吐。
             為了最大化資料管線（Data Pipeline）的並行吞吐量，可配置為 [ 2 * 物理核心數 ]。
             
        3. CPU (無加速晶片保底模式):
           - 最佳化解法：強制保留 1 個核心 [ 物理核心數 - 1 ] 專門供作業系統核心與背景系統調度使用，
             避免因模型計算導致 CPU 100% 鎖死而造成主機假死。
        """
        physical_cores = psutil.cpu_count(logical=False)
        if not physical_cores:
            physical_cores = 4  # 異常防禦邊界：若無法取得則給予保底 4 核心
        
        if device == "mps":
            return max(1, physical_cores)
        elif device == "cuda":
            return max(1, physical_cores * 2)
        else:
            return max(1, physical_cores - 1)