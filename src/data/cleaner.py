import re

class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        # 1. 移除非標準空白、換行與製表符
        text = re.sub(r"\s+", " ", text)
        # 2. 移除異常控制字元 (Control Characters)
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
        # 3. 轉為小寫並去除首尾空白
        return text.strip().lower()

    @staticmethod
    def tokenize_to_shingles(text: str, k: int = 5) -> set:
        """將文本切分為 k-shingling 集合，用於 Jaccard 相似度計算"""
        if len(text) < k:
            return {text} if text else set()
        return {text[i : i + k] for i in range(len(text) - k + 1)}