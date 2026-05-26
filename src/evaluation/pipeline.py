"""
Module Name: pipeline.py
Description: 基準測試自動化評估管線核心 (黃金標準答案完全體)
Author: Ace (Lead Architect)
"""

import asyncio
import aiohttp
from typing import List, Dict
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.judge import AsyncLLMJudge

class EvaluationPipeline:
    def __init__(self, provider_type: str, model: str, max_concurrent: int, timeout: float, api_key: str, api_url: str):
        self.metrics_calc = MetricsCalculator()
        self.judge = AsyncLLMJudge(
            provider_type=provider_type,
            model=model,
            max_concurrent=max_concurrent,
            timeout=timeout,
            api_key=api_key,
            api_url=api_url
        )

    async def execute(self, dataset: List[Dict[str, str]]) -> List[Dict]:
        print(f"🌀 啟動 Asyncio 事件迴圈，正在調度 {len(dataset)} 筆評估案例...")
        async with aiohttp.ClientSession() as session:
            results = []
            for case in dataset:
                res = await self._evaluate_case_stream(session, case)
                results.append(res)
                # 案例間冷卻 1.0 秒
                await asyncio.sleep(1.0)
            return results

    async def _evaluate_case_stream(self, session: aiohttp.ClientSession, case: Dict) -> Dict:
        case_id = case["id"]
        prompt = case["prompt"]
        base_res = case["baseline_response"]
        ft_res = case["finetuned_response"]
        target_res = case.get("target_response", "") # 🌟 引入黃金標準答案

        # 🚀 升級：分別以 target_response 作為 Reference 計算 ROUGE/BLEU 指標
        if target_res.strip():
            base_metrics = self.metrics_calc.compute_string_metrics(reference=target_res, hypothesis=base_res)
            ft_metrics = self.metrics_calc.compute_string_metrics(reference=target_res, hypothesis=ft_res)
        else:
            # Fallback 降級：若無標準答案，則退回原來的雙模型自我對抗模式
            base_metrics = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0, "em": 0.0}
            ft_metrics = self.metrics_calc.compute_string_metrics(reference=base_res, hypothesis=ft_res)

        # 序列化呼叫裁判，徹底拔除 429 轟炸併發
        judge_base = await self.judge.evaluate_single_case(session, prompt, base_res)

        # ⏳ 物理冷卻
        await asyncio.sleep(1.5)

        judge_ft = await self.judge.evaluate_single_case(session, prompt, ft_res)

        return {
            "id": case_id,
            "prompt": prompt,
            "base_metrics": base_metrics, # 🌟 拆分獨立指標流
            "ft_metrics": ft_metrics,
            "judge_baseline_score": judge_base["score"],
            "judge_baseline_reason": judge_base["reason"],
            "judge_finetuned_score": judge_ft["score"],
            "judge_finetuned_reason": judge_ft["reason"],
        }