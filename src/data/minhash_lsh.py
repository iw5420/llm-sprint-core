import numpy as np
import mmh3
from typing import List, Set, Dict, Tuple

class MinHashLSH:
    def __init__(self, num_perm: int = 128, num_bands: int = 32):
        self.num_perm = num_perm
        self.b = num_bands
        self.r = num_perm // num_bands
        
        if self.b * self.r != num_perm:
            raise ValueError("num_perm 必須能被 num_bands 整除")
            
        # 預先生成雜湊函數的隨機種子，確保雜湊置換矩陣的確定性
        np.random.seed(42)
        self.hash_seeds = np.random.randint(1, 1000000, size=self.num_perm, dtype=np.int32)
        # LSH 雜湊桶儲存結構: List[Dict[band_hash_tuple, List[doc_id]]]
        self.hashtables: List[Dict[Tuple, List[str]]] = [{} for _ in range(self.b)]

    def compute_minhash(self, shingles: Set[str]) -> np.ndarray:
        """計算單一文件的 MinHash 簽名向量"""
        signature = np.full(self.num_perm, fill_value=np.inf)
        if not shingles:
            return np.zeros(self.num_perm)
            
        for shingle in shingles:
            for i in range(self.num_perm):
                hv = mmh3.hash(shingle, seed=int(self.hash_seeds[i]))
                if hv < signature[i]:
                    signature[i] = hv
        return signature

    def insert_and_check_duplicate(self, doc_id: str, signature: np.ndarray) -> bool:
        """將簽名切分為 b 個 bands，並檢查是否與現有文件發生碰撞"""
        is_duplicate = False
        
        # 1. 查詢各個 Band 是否存在碰撞候選者
        for band_idx in range(self.b):
            start = band_idx * self.r
            end = start + self.r
            band_signature = tuple(signature[start:end])
            
            if band_signature in self.hashtables[band_idx]:
                is_duplicate = True  # 任一 Band 碰撞即視為高機率重複件

        # 2. 將當前文件寫入雜湊桶
        for band_idx in range(self.b):
            start = band_idx * self.r
            end = start + self.r
            band_signature = tuple(signature[start:end])
            if band_signature not in self.hashtables[band_idx]:
                self.hashtables[band_idx][band_signature] = []
            self.hashtables[band_idx][band_signature].append(doc_id)

        return is_duplicate