"""
Module Name: stream_loader.py
Description: 基於 IterableDataset 的生產級大文本 Streaming 流式加載核心。
Author: Ace (Lead Architect)
"""
import torch
from torch.utils.data import IterableDataset
from typing import Iterator, Dict, Callable
from src.data.packing import TokenPackingEngine

class StreamingPackingDataset(IterableDataset):
    """
    流式大文本打包傳送帶。
    """
    def __init__(self, file_path: str, mock_tokenizer: Callable[[str], list[int]], max_seq_len: int, eos_token_id: int):
        super().__init__()
        self.file_path = file_path
        self.tokenize_fn = mock_tokenizer
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        驅動流式發射器的核心迭代子。
        """
        # 初始化專屬的打包狀態機，確保每個 worker 擁有獨立狀態
        packing_engine = TokenPackingEngine(
            max_seq_len=self.max_seq_len,
            eos_token_id=self.eos_token_id
        )

        # 單向 I/O 文本流開啟
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 1. 進行 Tokenize（實際場景下此處可對齊接入任何生產級 Tokenizer）
                tokens = self.tokenize_fn(line)

                # 2. 餵入狀態機打包，並即時產出定長 Block
                for block in packing_engine.append_and_chunk(tokens):
                    # 自迴歸訓練標準：CPT 階段的 Inputs 與 Labels 完全是一致的（底層手寫 Loop 內會自動執行 Shift）
                    yield {"input_ids": block, "labels": block.clone()}

        # 3. 文本流讀取完畢，排空狀態機殘留數據
        for final_block in packing_engine.flush_remaining():
            yield {"input_ids": final_block, "labels": final_block.clone()}