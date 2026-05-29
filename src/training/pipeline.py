import torch
import torch.nn as nn
from src.training.lora_model import MockLinearWithLoRA
from src.training.hardware_profiler import RuntimeHardwareProfiler

class CustomTrainingLoopPipeline:
    def __init__(self, model: MockLinearWithLoRA, lr: float, grad_accum_steps: int, max_grad_norm: float, device: str):
        self.device = device
        self.model = model.to(self.device)
        self.accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm

        # 僅優化 requires_grad=True 的活躍低秩小表，死死守護 Base 唯讀矩陣
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: torch.utils.data.DataLoader, epoch_idx: int) -> float:
        # 1. 顯式宣示進入訓練狀態模式（開啟訓練計算圖）
        self.model.train()
        self.optimizer.zero_grad()

        running_loss = 0.0
        total_steps = len(dataloader)
        print(f"跑次 [{epoch_idx}] 核心計算開跑。總計 Mini-Batches 數量: {total_steps}")

        for step, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # 前向傳播（調用模型內部 forward 分流，外部無感知）
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 2. 梯度累積精確縮放：防止因為多步累積導致實質學習率被物理性放大
            loss_scaled = loss / self.accum_steps
            loss_scaled.backward()  # 手動發射反向傳播，數值累加於寫入緩衝區

            running_loss += loss.item()

            # 3. 判斷是否抵達累積邊界更新點
            if (step + 1) % self.accum_steps == 0 or (step + 1) == total_steps:
                # 4. 生產級防爆守護：在權重變更前，強行把暴衝的梯度數值砍回安全上限
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    max_norm=self.max_grad_norm
                )

                # 實體權重踩踏前進，並隨即清空核心緩衝區
                self.optimizer.step()
                self.optimizer.zero_grad()

                if step % (self.accum_steps * 2) == 0:
                    print(f"   [Step {step+1}/{total_steps}] 實時 Step-Loss: {loss.item():.6f}")
                    RuntimeHardwareProfiler.log_memory_snapshot(self.device)

        return running_loss / total_steps