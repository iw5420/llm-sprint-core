"""
Module Name: pipeline.py
Description: DPO 偏好對齊微調傳送帶控制管線核心。
Author: Ace (Lead Architect)
"""
import torch
import torch.nn as nn
from src.alignment.dpo_loss import DirectPreferenceLoss

class DPOTrainingPipeline:
    def __init__(self, policy_model: nn.Module, ref_model: nn.Module, lr: float, beta: float, max_grad_norm: float, device: str):
        self.device = device

        # Policy Model：當前的微調核心，僅針對其內部解鎖且帶有 requires_grad=True 的 LoRA 參數進行優化
        self.policy_model = policy_model.to(self.device)

        # Reference Model：對比大庫，必須在全生命週期鎖死梯度，充當不變的黃金常駐參考錨點
        self.ref_model = ref_model.to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.policy_model.parameters()), lr=lr)
        self.dpo_criterion = DirectPreferenceLoss(beta=beta)
        self.max_grad_norm = max_grad_norm

    def train_step(self, batch: dict) -> dict:
        """
        【函數任務：執行單步 DPO 偏好優化傳送帶】
        精心控制兩套模型的 forward 分流，徹底封鎖 Reference 端的寫入快取記憶體。
        """
        self.policy_model.train()
        self.optimizer.zero_grad()

        # 1. 將輸入數據安全搬移至指定運算晶片上
        c_inputs = batch["chosen_inputs"].to(self.device)
        c_labels = batch["chosen_labels"].to(self.device)
        r_inputs = batch["rejected_inputs"].to(self.device)
        r_labels = batch["rejected_labels"].to(self.device)

        # 2. 驅動當前微調模型 (Policy Forward)
        policy_chosen_logits = self.policy_model(c_inputs)
        policy_rejected_logits = self.policy_model(r_inputs)

        # 3. 驅動凍結參考模型 (Reference Forward) -> 必須使用 torch.no_grad() 阻斷計算圖，防範內存爆炸
        with torch.no_grad():
            ref_chosen_logits = self.ref_model(c_inputs)
            ref_rejected_logits = self.ref_model(r_inputs)

        # 4. 呼叫純手寫 DPO 損失引擎，取得核心數值
        loss, chosen_rewards, rejected_rewards = self.dpo_criterion(
            policy_chosen_logits, c_labels,
            policy_rejected_logits, r_labels,
            ref_chosen_logits, ref_chosen_labels=c_labels, # 標籤恆等比對
            ref_rejected_logits=ref_rejected_logits, ref_rejected_labels=r_labels
        )

        # 5. 反向傳播與防爆安全裁剪
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.policy_model.parameters()), max_norm=self.max_grad_norm)

        self.optimizer.step()

        # 6. 計算隱式獎勵差（Delta Reward），做為觀測指標
        delta_reward = (chosen_rewards - rejected_rewards).mean().item()

        return {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "delta_reward": delta_reward
        }