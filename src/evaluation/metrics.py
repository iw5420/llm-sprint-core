import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class MetricsCalculator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth_fn = SmoothingFunction().method1

    def compute_string_metrics(self, reference: str, hypothesis: str) -> dict:
        if not reference.strip() or not hypothesis.strip():
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0, "em": 0.0}

        scores = self.scorer.score(reference, hypothesis)
        ref_tokens = list(reference.strip())
        hyp_tokens = list(hypothesis.strip())
        bleu_score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=self.smooth_fn)
        em_score = 1.0 if reference.strip() == hypothesis.strip() else 0.0

        return {
            "rouge1": scores['rouge1'].fmeasure * 100,
            "rouge2": scores['rouge2'].fmeasure * 100,
            "rougeL": scores['rougeL'].fmeasure * 100,
            "bleu": bleu_score * 100,
            "em": em_score * 100
        }