"""
Module Name: test_cpt_packing.py
Description: 工業級 Packing 狀態機與流式大文本傳送帶物理邊界單元測試。
Author: Ace (Lead Architect)
"""
import pytest
import torch
import os
from src.data.packing import TokenPackingEngine
from src.data.stream_loader import StreamingPackingDataset

def test_packing_engine_exact_chunking():
    """
    🔥 測試一：驗證 Packing 狀態機能否精確執行定長切分，且無遺漏地緩衝殘餘數據。
    """
    max_seq_len = 4
    eos_token_id = 99
    engine = TokenPackingEngine(max_seq_len=max_seq_len, eos_token_id=eos_token_id)

    # 塞入 2 個 token，加上 [EOS] 後長度為 3。不足 4，此時應不吐出任何 Block
    blocks = list(engine.append_and_chunk([1, 2]))
    assert len(blocks) == 0
    assert engine.residual_buffer == [1, 2, 99]

    # 再塞入 2 個 token，加上 [EOS] 後總緩衝區變為 [1, 2, 99, 3, 4, 99]
    # 長度為 6，此時應吐出一個長度為 4 的 Block，並剩下 [4, 99]
    blocks_two = list(engine.append_and_chunk([3, 4]))
    assert len(blocks_two) == 1
    assert torch.equal(blocks_two[0], torch.tensor([1, 2, 99, 3], dtype=torch.long))
    assert engine.residual_buffer == [4, 99]

    # 執行強制清倉，剩餘的 [4, 99] 應被右側 Padding 補齊成 [4, 99, 99, 99] 倒出
    flush_blocks = list(engine.flush_remaining())
    assert len(flush_blocks) == 1
    assert torch.equal(flush_blocks[0], torch.tensor([4, 99, 99, 99], dtype=torch.long))
    assert len(engine.residual_buffer) == 0

def test_streaming_packing_dataset_pipeline(tmp_path):
    """
    🔥 測試二：建立虛擬暫存文本，檢驗 StreamingPackingDataset 全鏈路流式吞吐。
    """
    # 建立暫存測試語料檔
    test_file = tmp_path / "dummy_corpus.txt"
    test_file.write_text("hello world\nllm sprint core\n", encoding="utf-8")

    def dummy_tokenizer(text: str) -> list[int]:
        # 簡單映射：每個字轉換成固定代碼
        return [10, 20]

    # max_seq_len = 3, eos = 2. 總共兩行，每行產出 2 個 token
    # 第一行 tokens + eos = [10, 20, 2] -> 剛好滿 3, 吐出 Block 1
    # 第二行 tokens + eos = [10, 20, 2] -> 剛好滿 3, 吐出 Block 2
    dataset = StreamingPackingDataset(
        file_path=str(test_file),
        mock_tokenizer=dummy_tokenizer,
        max_seq_len=3,
        eos_token_id=2
    )

    data_list = list(dataset)
    assert len(data_list) == 2
    assert torch.equal(data_list[0]["input_ids"], torch.tensor([10, 20, 2], dtype=torch.long))
    assert torch.equal(data_list[1]["input_ids"], torch.tensor([10, 20, 2], dtype=torch.long))