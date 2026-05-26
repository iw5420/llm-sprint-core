from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal
from src.config.hardware import RuntimeHardwareDetector

class EnvConfigSchema(BaseModel):
    """環境憑證安全防禦區塊"""
    provider_type: Literal["openai", "gemini", "vllm"] = Field(..., alias="PROVIDER_TYPE")
    provider_api_key: str = Field(..., alias="PROVIDER_API_KEY")
    eval_api_url: str = Field(..., alias="EVAL_API_URL")

    @model_validator(mode="after")
    def validate_provider_routing(self) -> "EnvConfigSchema":
        url = self.eval_api_url.lower()
        if self.provider_type == "gemini" and "openai.com" in url:
            raise ValueError("環境組態衝突：PROVIDER_TYPE 設為 gemini，但 EVAL_API_URL 卻指向 OpenAI 官方端點！")
        return self

class HyperParametersSchema(BaseModel):
    """跨模組演算法與實驗超參數安全邊界校驗"""
    r: int = Field(..., description="LoRA Rank 階數")
    alpha: int = Field(..., description="LoRA 縮放因子常數")
    lr: float = Field(..., description="優化器初始學習率")
    max_seq_len: int = Field(..., description="最大上下文 Token 輸入長度")
    num_perm: int = Field(..., ge=1, description="MinHash 雜湊置換次數")
    num_bands: int = Field(..., ge=1, description="LSH 雜湊桶區段數")
    max_concurrent_tasks: int = Field(..., ge=1, le=100, description="最大非同步並發 API 數量")
    api_timeout_seconds: float = Field(..., ge=5.0, description="外部 API 呼叫安全超時秒數")
    eval_model: str = Field(..., description="外部通用裁判模型的實體名稱")

    @field_validator("r")
    @classmethod
    def validate_rank(cls, v: int) -> int:
        if v <= 0 or (v & (v - 1)) != 0:
            raise ValueError(f"LoRA Rank 必須為大於 0 的 2 的冪次方，當前非法輸入值: {v}")
        return v

    @field_validator("lr")
    @classmethod
    def validate_lr(cls, v: float) -> float:
        if not (1e-6 <= v <= 1e-2):
            raise ValueError(f"學習率 lr 超出安全邊界限制 [1e-6, 1e-2]，當前非法輸入值: {v}")
        return v

    @field_validator("max_seq_len")
    @classmethod
    def validate_seq_len(cls, v: int) -> int:
        if v <= 0 or v > 8192:
            raise ValueError(f"不支援的上下文長度限制，當前輸入值: {v}")
        return v

    @model_validator(mode="after")
    def validate_lsh_math(self) -> "HyperParametersSchema":
        if self.num_perm % self.num_bands != 0:
            raise ValueError(f"LSH 配置規格錯誤：num_perm ({self.num_perm}) 必須能被 num_bands ({self.num_bands}) 整除！")
        return self

class AppConfigSchema(BaseModel):
    """全局唯一統合 Fact 配置物件 (SSOT Core)"""
    project_name: str = Field(..., description="專案識別名稱")
    debug: bool = Field(default=True, description="除錯模式開關")
    hparams: HyperParametersSchema = Field(..., description="巢狀超參數物件")
    env: EnvConfigSchema = Field(..., description="巢狀環境變數物件")
    device: Literal["mps", "cuda", "cpu"] = "cpu"
    optimal_threads: int = Field(default=4, ge=1)

    @model_validator(mode="before")
    @classmethod
    def inject_runtime_hardware(cls, data: dict) -> dict:
        if isinstance(data, dict):
            detected_device = RuntimeHardwareDetector.detect_device()
            optimal_threads = RuntimeHardwareDetector.calculate_optimal_threads(detected_device)
            data["device"] = detected_device
            data["optimal_threads"] = optimal_threads
        return data