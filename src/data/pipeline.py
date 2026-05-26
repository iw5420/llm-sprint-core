import os
from multiprocessing import Pool
from typing import List, Dict, Tuple
import numpy as np
from src.data.cleaner import TextCleaner
from src.data.minhash_lsh import MinHashLSH

def _worker_clean_and_minhash(task: Tuple[str, str, int, int]) -> Tuple[str, str, list]:
    """Worker 進程：無狀態 CPU 密集計算（清洗 + 生成 MinHash 簽名）"""
    # 🌟 核心修正點：確保此處為 4 元素解包，與 execute 中的 tasks 規格完全一致
    doc_id, raw_text, num_perm, num_bands = task

    cleaned_text = TextCleaner.clean(raw_text)
    shingles = TextCleaner.tokenize_to_shingles(cleaned_text, k=5)

    # 🌟 顯式注入組態參數，解除硬編碼耦合
    lsh_helper = MinHashLSH(num_perm=num_perm, num_bands=num_bands)
    signature = lsh_helper.compute_minhash(shingles).tolist()

    return doc_id, cleaned_text, signature

class DataCurationPipeline:
    def __init__(self, num_workers: int, num_perm: int, num_bands: int):
        self.num_workers = num_workers
        self.num_perm = num_perm
        self.num_bands = num_bands

    def execute(self, dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # 主進程打包 4 元素元組發送
        tasks = [(doc["id"], doc["text"], self.num_perm, self.num_bands) for doc in dataset]

        print(f"🌀 啟動 {self.num_workers} 個工作進程進行文本清洗與 MinHash 簽名計算...")
        with Pool(processes=self.num_workers) as pool:
            processed_results = pool.map(_worker_clean_and_minhash, tasks)

        print("🔍 進入主進程 LSH 桶聚合與去重階段...")
        # 主進程聚合階段同步使用強型態規格
        lsh = MinHashLSH(num_perm=self.num_perm, num_bands=self.num_bands)
        unique_documents = []

        for doc_id, cleaned_text, sig_list in processed_results:
            if not cleaned_text:
                continue
            signature = np.array(sig_list)
            is_dup = lsh.insert_and_check_duplicate(doc_id, signature)

            if not is_dup:
                unique_documents.append({"id": doc_id, "text": cleaned_text})

        return unique_documents