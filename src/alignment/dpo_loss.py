"""
Module Name: dpo_loss.py
Description: 純手寫 DPO (Direct Preference Optimization) 損失函數與 token-level 對數機率提取核心。
             完全廢棄外部高階框架，純採用 PyTorch Tensor 矩陣運算。
Author: Ace (Lead Architect)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectPreferenceLoss(nn.Module):
    """
    純手寫 DPO 損失函數引擎。
    【核心架構】
    傳統人類偏好對齊（如 RLHF）需要額外訓練一個獎勵模型（Reward Model），開銷巨大。
    DPO 的精妙之處在於透過數學變換，直接利用「當前微調模型」與「原始參考模型」的輸出機率差，
    隱式地拉開 Chosen（人類滿意的回答）與 Rejected（人類討厭的回答）的獎勵分佈。
    """
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def _get_batch_logps(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        r"""
        【函數任務：Token-Level 矩陣對數機率提取器】

        【剛接觸 LLM 者必讀指南】：
        1. logits 的維度通常是 [Batch_Size, Seq_Len, Vocab_Size]，代表模型對每個位置預測下一個字的「未歸一化機率」。
        2. labels 的維度是 [Batch_Size, Seq_Len]，是真實的文字 Token ID。
        3. 我們必須計算模型預測出這些真實文字的「累積對數機率」。

        數學公式推導：
        $$\log P(\mathbf{y} \mid \mathbf{x}) = \sum_{t=1}^{T} \log P(y_t \mid \mathbf{x}, y_{<t})$$
        """
        # 狀態防禦：確保 logits 與 labels 的批次大小與序列長度卡緊對齊
        assert logits.shape[0] == labels.shape[0]
        assert logits.shape[1] == labels.shape[1]

        # 1. 透過 LogSoftmax 將 logits 轉換為標準對數機率，維度保持 [B, S, V]
        log_probs = F.log_softmax(logits, dim=-1)

        # 2. 使用 gather 算子，依據真實 labels 的 Token ID，從 Vocab_Size 維度中榨取出模型當前預測該字的位置機率
        # labels.unsqueeze(-1) 將維度變為 [B, S, 1]，抽樣後再 squeeze(-1) 變回 [B, S]
        per_token_logps = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        # 3. 沿著序列長度（dim=-1）將每個字元預測的對數機率加總（Log 的加總等價於實際機率的相乘），最終回傳 [Batch_Size]
        return per_token_logps.sum(dim=-1)

    def forward(self,
                policy_chosen_logits: torch.Tensor, policy_chosen_labels: torch.Tensor,
                policy_rejected_logits: torch.Tensor, policy_rejected_labels: torch.Tensor,
                ref_chosen_logits: torch.Tensor, ref_chosen_labels: torch.Tensor,
                ref_rejected_logits: torch.Tensor, ref_rejected_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        【函數任務：執行 DPO 損失計算與獎勵隱式對齊】

        數學核心公式完全體：
        $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(\mathbf{x}, \mathbf{y}_w, \mathbf{y}_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(\mathbf{y}_w \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_w \mid \mathbf{x})} - \beta \log \frac{\pi_\theta(\mathbf{y}_l \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_l \mid \mathbf{x})} \right) \right]$$

        回傳值:
            losses: 標量張量 (梯度下降核心)
            chosen_rewards: 微調模型對於滿意回答的隱式獎勵數值
            rejected_rewards: 微調模型對於討厭回答的隱式獎勵數值
        """
        # 步驟一：分別榨取出「當前微調模型 (Policy)」對於 Chosen/Rejected 文本的總體對數機率
        policy_chosen_logps = self._get_batch_logps(policy_chosen_logits, policy_chosen_labels)
        policy_rejected_logps = self._get_batch_logps(policy_rejected_logits, policy_rejected_labels)

        # 步驟二：分別榨取出「原始凍結模型 (Reference)」對於 Chosen/Rejected 文本的總體對數機率
        ref_chosen_logps = self._get_batch_logps(ref_chosen_logits, ref_chosen_labels)
        ref_rejected_logps = self._get_batch_logps(ref_rejected_logits, ref_rejected_labels)

        # 步驟三：計算當前模型相較於原始模型的機率增長值 (Log 空間相減等價於機率相除)
        # 這代表模型在微調過程中，對這兩份資料的偏好改變率
        policy_ref_chosen_ratio = policy_chosen_logps - ref_chosen_logps
        policy_ref_rejected_ratio = policy_rejected_logps - ref_rejected_logps

        # 步驟四：計算隱式獎勵 (Implicit Reward)。這一步是 DPO 的數學精髓
        # 乘以 beta 係數後，可以直接做為衡量模型偏好程度的純量指標
        chosen_rewards = self.beta * policy_ref_chosen_ratio
        rejected_rewards = self.beta * policy_ref_rejected_ratio

        # 步驟五：構造拉開兩者差距的損失矩陣
        # 我們的目標是讓 chosen_rewards 遠遠大於 rejected_rewards
        logits_diff = chosen_rewards - rejected_rewards

        # 使用 F.logsigmoid 逼近優化邊界。當 logits_diff 越大，logsigmoid 就越接近 0 (損失極小)
        # 加上負號即可進行最小化梯度下降
        losses = -F.logsigmoid(logits_diff).mean()

        return losses, chosen_rewards.detach(), rejected_rewards.detach()