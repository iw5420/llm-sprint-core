"""
Module Name: packing.py
Description: 工業級文本無損打包（Packing）引擎，破除 Padding 算力浪費。
Author: Ace (Lead Architect)
"""
import torch
from typing import Generator, List

class TokenPackingEngine:
    """
    高效文本打包狀態機。
    在記憶體中維護一個動態常駐緩衝區，專門執行跨文本的拼接與精確定長切分。
    """
    def __init__(self, max_seq_len: int, eos_token_id: int):
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self.residual_buffer: List[int] = []  # 歷史殘留 Token 緩衝區

    def append_and_chunk(self, token_ids: List[int]) -> Generator[torch.Tensor, None, None]:
        """
        將新流入的單篇短文本 Token 序列，拼接到狀態機中，並吐出所有達標的定長 Block。
        """
        # 1. 注入當前文本，並在尾端物理性追加工業級對齊隔離符 [EOS]
        self.residual_buffer.extend(token_ids + [self.eos_token_id])

        # 2. 循環提取，直到剩餘量不足以湊滿一個標準的物理區塊長度 (max_seq_len)
        while len(self.residual_buffer) >= self.max_seq_len:
            # 彈出前項定長窗口
            chunk = self.residual_buffer[:self.max_seq_len]
            self.residual_buffer = self.residual_buffer[self.max_seq_len:]

            # 轉化為高效張量傳回
            yield torch.tensor(chunk, dtype=torch.long)

    def flush_remaining(self) -> Generator[torch.Tensor, None, None]:
        """
        【清倉閘門】當所有外部文本流乾涸時，強制將殘留的剩餘數據打包輸出。
        此處僅需進行一次性 Padding，將算力耗損降至全生命週期的最低點。
        """
        if len(self.residual_buffer) > 0:
            pad_len = self.max_seq_len - len(self.residual_buffer)
            # 僅在最終尾端補足殘餘空間
            final_chunk = self.residual_buffer + [self.eos_token_id] * pad_len
            self.residual_buffer = []
            yield torch.tensor(final_chunk, dtype=torch.long)