from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal
from src.config.hardware import RuntimeHardwareDetector

class HyperParametersSchema(BaseModel):
    """
    定義 LoRA 訓練的核心超參數，並進行嚴格的數學與安全邊界校驗。
    """
    r: int = Field(..., description="LoRA 內部的低秩矩陣 Rank 階數")
    alpha: int = Field(..., description="LoRA 縮放因子常數 Scaling Factor")
    lr: float = Field(..., description="優化器初始學習率")
    max_seq_len: int = Field(..., description="最大上下文 Token 輸入長度")
    num_perm: int = Field(default=128, description="MinHash 雜湊置換次數")

    @field_validator("r")
    @classmethod
    def validate_rank(cls, v: int) -> int:
        # 位元運算技巧：確保 Rank 為 2 的冪次方，這對底層計算效能優化至關重要
        if v <= 0 or (v & (v - 1)) != 0:
            raise ValueError(f"LoRA Rank 必須為大於 0 的 2 的冪次方，當前非法輸入值: {v}")
        return v

    @field_validator("lr")
    @classmethod
    def validate_lr(cls, v: float) -> float:
        # 防止學習率設得太極端導致訓練發散 (梯度爆炸/消失)
        if not (1e-6 <= v <= 1e-2):
            raise ValueError(f"學習率 lr 超出安全邊界限制 [1e-6, 1e-2]，當前非法輸入值: {v}")
        return v

    @field_validator("max_seq_len")
    @classmethod
    def validate_seq_len(cls, v: int) -> int:
        # 根據顯存/記憶體容量限制，防止 OOM (Out of Memory)
        if v <= 0 or v > 8192:
            raise ValueError(f"不支援的上下文長度限制，當前輸入值: {v}")
        return v


class AppConfigSchema(BaseModel):
    """
    應用程式主配置，整合硬體感知功能，達成「環境無關性」的設計目標。
    """
    project_name: str = Field(..., description="專案識別名稱")
    debug: bool = Field(default=True, description="除錯模式開關")
    hparams: HyperParametersSchema = Field(..., description="巢狀超參數物件")
    
    # 以下兩個欄位由 model_validator 自動注入，手動輸入會被 Runtime 偵測覆蓋
    device: Literal["mps", "cuda", "cpu"] = "cpu"
    optimal_threads: int = Field(default=4, ge=1)

    @model_validator(mode="before")
    @classmethod
    def inject_runtime_hardware(cls, data: dict) -> dict:
        """
        動態注入邏輯：在驗證數據前，自動偵測目前硬體（MPS/CUDA/CPU）並計算最佳執行緒。
        目的：確保同一套代碼在 Mac(M4 Pro) 或 Linux GPU Server 都能以最優配置啟動。
        """
        if isinstance(data, dict):
            # 調用先前的 RuntimeHardwareDetector 進行實時偵測
            detected_device = RuntimeHardwareDetector.detect_device()
            optimal_threads = RuntimeHardwareDetector.calculate_optimal_threads(detected_device)
            
            # 將偵測結果強制寫入配置字典中
            data["device"] = detected_device
            data["optimal_threads"] = optimal_threads
            
        return data