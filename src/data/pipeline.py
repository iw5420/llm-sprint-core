import os
from multiprocessing import Pool
from typing import List, Dict, Tuple
import numpy as np
from src.data.cleaner import TextCleaner
from src.data.minhash_lsh import MinHashLSH

def _worker_clean_and_minhash(task: Tuple[str, str, int]) -> Tuple[str, str, list]:
    """Worker 進程：無狀態 CPU 密集計算（清洗 + 生成 MinHash 簽名）"""
    doc_id, raw_text, num_perm = task
    cleaned_text = TextCleaner.clean(raw_text)
    shingles = TextCleaner.tokenize_to_shingles(cleaned_text, k=5)
    
    lsh_helper = MinHashLSH(num_perm=num_perm)
    signature = lsh_helper.compute_minhash(shingles).tolist() # 轉成 list 以利 pickle 序列化
    
    return doc_id, cleaned_text, signature

class DataCurationPipeline:
    def __init__(self, num_workers: int, num_perm: int = 128):
        self.num_workers = num_workers
        self.num_perm = num_perm

    def execute(self, dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
        tasks = [(doc["id"], doc["text"], self.num_perm) for doc in dataset]
        
        # 1. 並行化無狀態計算階段
        print(f"🌀 啟動 {self.num_workers} 個工作進程進行文本清洗與 MinHash 簽名計算...")
        with Pool(processes=self.num_workers) as pool:
            processed_results = pool.map(_worker_clean_and_minhash, tasks)

        # 2. 單執行緒有狀態 LSH 篩選階段 (避免多進程鎖同步開銷)
        print("🔍 進入主進程 LSH 桶聚合與去重階段...")
        lsh = MinHashLSH(num_perm=self.num_perm)
        unique_documents = []

        for doc_id, cleaned_text, sig_list in processed_results:
            if not cleaned_text:
                continue
            signature = np.array(sig_list)
            is_dup = lsh.insert_and_check_duplicate(doc_id, signature)
            
            if not is_dup:
                unique_documents.append({"id": doc_id, "text": cleaned_text})

        return unique_documents