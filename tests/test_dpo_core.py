"""
Module Name: test_dpo_core.py
Description: 模組五手寫 DPO 損失與對數機率核心之數學恆等 TDD 測試（精度與 Softmax 破口平反完全體）。
Author: Ace (Lead Architect)
"""
import pytest
import torch
import torch.nn.functional as F
from src.alignment.dpo_loss import DirectPreferenceLoss

def test_get_batch_logps_mathematical_identity():
    """【測試目標】驗證 Token-Level 對數機率提取器之多維 gather 運算的精確度與恆等性"""
    dpo_criterion = DirectPreferenceLoss(beta=0.1)

    # 建立一個極小、可手動追蹤計算的機率矩陣
    # Batch_Size=1, Seq_Len=2, Vocab_Size=3
    mock_logits = torch.tensor([[[1.0, 2.0, 0.0],
                                 [0.0, 3.0, 1.0]]], dtype=torch.float32)
    mock_labels = torch.tensor([[1, 2]], dtype=torch.long) # 真實標籤

    # 呼叫手寫核心提取器
    calculated_logps = dpo_criterion._get_batch_logps(mock_logits, mock_labels)

    assert calculated_logps.dim() == 1
    assert calculated_logps.shape[0] == 1

    # 🌟 精度校正：對齊 PyTorch 核心高精度 C++ 算子運算之實體真值 -2.5775
    assert torch.allclose(calculated_logps, torch.tensor([-2.5775]), atol=1e-3)


def test_dpo_loss_preference_拉開邊界收斂性():
    """【測試目標】驗證當 Chosen 的機率顯著大於 Rejected 時，DPO 損失函數必須展現出收斂且極小的特性"""
    dpo_criterion = DirectPreferenceLoss(beta=0.1)

    B, S, V = 2, 4, 10
    labels = torch.randint(0, V, (B, S))

    # ----------------------------------------------------------------------
    # 1. 模擬理想狀態：微調模型高度精準。
    # 核心防禦：不能全矩陣填滿常數！必須使用 gather/scatter 讓 True Labels 的位置分值極高，其餘錯字極低。
    # ----------------------------------------------------------------------
    policy_c_logits = torch.zeros((B, S, V))
    # 將真實 labels 位置的 logit 頂到 +10.0（確保 Chosen 概率極大）
    policy_c_logits.scatter_(dim=-1, index=labels.unsqueeze(-1), value=10.0)

    policy_r_logits = torch.zeros((B, S, V))
    # 將真實 labels 位置的 logit 踩到 -10.0（確保 Rejected 概率極小）
    policy_r_logits.scatter_(dim=-1, index=labels.unsqueeze(-1), value=-10.0)

    # 參考模型保持全零中立
    ref_logits = torch.zeros((B, S, V))

    loss_ideal, chosen_reward_i, rejected_reward_i = dpo_criterion(
        policy_c_logits, labels, policy_r_logits, labels,
        ref_logits, labels, ref_logits, labels
    )

    # ----------------------------------------------------------------------
    # 2. 模擬糟糕狀態：微調模型對於 Chosen 與 Rejected 傻傻分不清（Logits 全面中立一致）
    # ----------------------------------------------------------------------
    policy_neutral = torch.zeros((B, S, V))
    loss_bad, chosen_reward_b, rejected_reward_b = dpo_criterion(
        policy_neutral, labels, policy_neutral, labels,
        ref_logits, labels, ref_logits, labels
    )

    print(f"\n[TDD 邊界對比觀測] 理想狀態 Loss = {loss_ideal.item():.6f} | 糟糕狀態 Loss = {loss_bad.item():.6f}")
    print(f"[TDD 獎勵觀測] Chosen 隱式獎勵: {chosen_reward_i.mean().item():.4f} | Rejected 隱式獎勵: {rejected_reward_i.mean().item():.4f}")

    # ⚖️ 批判性斷言驗證
    # 斷言一：經過 scatter_ 破除平移不變性後，理想狀態的 DPO Loss 必須顯著收斂並小於糟糕狀態 (ln(2) = 0.6931)
    assert loss_ideal.item() < loss_bad.item()

    # 斷言二：理想狀態下，Chosen 獲得的隱式獎勵必須大於 Rejected 獲得的獎勵
    assert chosen_reward_i.mean().item() > rejected_reward_i.mean().item()

    # 斷言三：在完全均勻的糟糕狀態下，DPO 隱式獎勵差（Delta）應趨近於 0
    assert torch.allclose(chosen_reward_b, rejected_reward_b, atol=1e-5)