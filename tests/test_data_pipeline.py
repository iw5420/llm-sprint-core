"""
Module Name: test_data_pipeline.py
Description: 模組二資料工程去重管線單元測試。
             已補齊 num_bands 參數簽章，確保與全域 SSOT 組態強型態安全對齊。
Author: Ace (Lead Architect)
"""

import pytest
from src.data.cleaner import TextCleaner
from src.data.minhash_lsh import MinHashLSH
from src.data.pipeline import DataCurationPipeline

def test_text_cleaner_edge_cases():
    """
    【測試目標】驗證文本標準化清洗核心（TextCleaner）的防禦力與邊界條件處理。
    【測試焦點】確保空值不崩潰、惡意空白被收斂、隱藏的異常控制字元被徹底過濾。
    """
    # 測試 A: 確保傳入空字串時，系統能安全回傳空字串，不會引發非預期錯誤
    assert TextCleaner.clean("") == ""

    # 測試 B: 確保型態錯誤（傳入 None）時，具備強健的防禦邏輯，安全回傳空字串
    assert TextCleaner.clean(None) == ""

    # 測試 C: 傳入含有極端換行 (\n)、製表符 (\t)、連續空格與惡意控制字元 (\x00, \x7F) 的髒亂大文本
    dirty_text = "  LLM\n\n\tOps  \x00Lifecycle\x7F "
    # 預期結果: 必須去除非法字元、將連續空白收斂為單一空格、文字全部轉小寫並去除首尾空白
    assert TextCleaner.clean(dirty_text) == "llm ops lifecycle"


def test_minhash_lsh_deduplication():
    """
    【測試目標】驗證純手寫 MinHash + LSH 引擎的「模糊去重（Fuzzy Deduplication）」核心邏輯。
    【測試焦點】確保演算法能抓出「只差一個標點符號」的高相似文本，同時放行「完全無關」的正常文本。
    """
    # 初始化去重引擎，設定 128 次雜湊置換與 32 個 LSH 雜湊桶
    lsh = MinHashLSH(num_perm=128, num_bands=32)

    # 準備三篇測試文本，並切分為 k-shingling 集合（字元片段滑動視窗）
    doc1 = TextCleaner.tokenize_to_shingles("this is a high concurrency pipeline for llm data")
    doc2 = TextCleaner.tokenize_to_shingles("this is a high concurrency pipeline for llm data!") # 僅尾端多一個驚嘆號
    doc3 = TextCleaner.tokenize_to_shingles("completely different text content for deep learning") # 全新無關主題

    # 計算各個文件的 MinHash 簽名向量（將龐大的字元片段壓縮為 128 維的整數簽名）
    sig1 = lsh.compute_minhash(doc1)
    sig2 = lsh.compute_minhash(doc2)
    sig3 = lsh.compute_minhash(doc3)

    # 斷言 1: doc1 是系統收錄的第一筆資料，此時資料庫為空，預期回傳 False（不是重複件）
    assert lsh.insert_and_check_duplicate("id1", sig1) is False

    # 斷言 2: doc2 與 doc1 僅差一個驚嘆號，Jaccard 相似度極高。LSH 分桶應發生雜湊碰撞，預期回傳 True（成功攔截高度相似件）
    assert lsh.insert_and_check_duplicate("id2", sig2) is True

    # 斷言 3: doc3 談論深度學習，是全新獨立內容。分桶時不應與 doc1/doc2 發生碰撞，預期回傳 False（判定為乾淨的新文本）
    assert lsh.insert_and_check_duplicate("id3", sig3) is False


def test_pipeline_end_to_end():
    """
    【測試目標】驗證整合型多進程數據管線（DataCurationPipeline）的端到端（E2E）運作鏈路。
    【測試焦點】模擬真實語料庫輸入，確保 Multiprocessing 多核心調度清洗、簽名、LSH 過濾聚合能完美協同。
    """
    # 建立一組模擬的私有文本語料庫（內含重複件、空字串件以及正常件）
    mock_data = [
        {"id": "1", "text": "Parallel computing with Python multiprocessing is efficient."},
        {"id": "2", "text": "Parallel computing with Python multiprocessing is efficient!!!"}, # 應被過濾的相似重複件
        {"id": "3", "text": ""},                                                                 # 應被過濾的無效空件
        {"id": "4", "text": "Isolated data infrastructure engineering."}                         # 應被保留的正常件
    ]

    # 🌟 修正對齊：初始化管線時，必須完全對齊主架構，配置 num_bands=32，確保 LSH 雜湊桶分桶完備
    pipeline = DataCurationPipeline(num_workers=2, num_perm=128, num_bands=32)

    # 執行端到端大文本吞吐清洗與去重管線
    curated_dataset = pipeline.execute(mock_data)

    # 斷言 A: 經過多核心清洗與 LSH 模糊去重後，原始 4 筆數據預期最終只留下 2 筆乾淨、有效的唯一文本
    assert len(curated_dataset) == 2

    # 斷言 B: 收集留存下來的文件 ID，確保最終留下的是核心功能完好的 "id_1" 與全新內容的 "id_4"
    remaining_ids = {doc["id"] for doc in curated_dataset}
    assert "1" in remaining_ids and "4" in remaining_ids