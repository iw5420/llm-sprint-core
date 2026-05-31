"""
Module Name: cpt_loop.py
Description: 手寫 Continued Pre-Training 核心自迴歸（Causal LM）訓練迴路。
             加入動態 Embedding 投影層，修正 Token ID 到特徵向量的維度破口。
Author: Ace (Lead Architect)
"""
import torch
import torch.nn as nn

class CPTTrainingPipeline:
    def __init__(self, model: nn.Module, lr: float, max_grad_norm: float, device: str):
        self.device = device
        self.model = model.to(self.device)

        # 🌟 增量防禦：建立 Token 嵌入層，將 [B, S] 的 Token ID 物理性投影至 [B, S, 128] 的特徵定址空間
        # 256 為 Vocab Size, 128 為模型的 in_features
        self.embedding = nn.Embedding(num_embeddings=256, embedding_dim=128).to(self.device)

        # 將 embedding 參數納入優化器，對齊全量預訓練行為
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.embedding.parameters()),
            lr=lr
        )

        self.criterion = nn.CrossEntropyLoss()
        self.max_grad_norm = max_grad_norm

    def train_step(self, batch: dict) -> float:
        """
        單步 CPT 自迴歸因果遮罩矩陣演練步履。
        """
        self.model.train()
        self.embedding.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # 1. 🌟 維度對齊閘門：將 [2, 2048] 轉化為 [2, 2048, 128]
        hidden_states = self.embedding(input_ids)

        # 2. 前向傳播 (此時 hidden_states 進入 model 將呈現完美的 (2x2048) x 128 與 128x256 矩陣乘法)
        logits = self.model(hidden_states) # 預期輸出維度: [Batch_Size, Seq_Len, Vocab_Size]

        # 3. 執行 Causal LM 的標準位移（Shift 算子）
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 4. 展平矩陣以適應 CrossEntropyLoss 的二維定址規格
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # 5. 反向傳播與防爆裁剪
        loss.backward()

        # 收集所有需要裁剪的參數
        all_params = list(self.model.parameters()) + list(self.embedding.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.max_grad_norm)

        self.optimizer.step()

        return loss.item()