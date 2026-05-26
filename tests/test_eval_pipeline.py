"""
Module Name: test_eval_pipeline.py
Description: 評估管線核心之非同步單元測試與極端邊界防禦機制。
Author: Ace (Lead Architect)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from src.evaluation.pipeline import EvaluationPipeline

@pytest.mark.asyncio
async def test_pipeline_successful_flow_with_ground_truth():
    """
    1. 驗證標準黃金數據流：
       當 mock 數據中包含 target_response 時，
       管線必須精確計算出獨立的 base_metrics 與 ft_metrics，且成功序列化取得裁判分數。
    """
    mock_dataset = [{
        "id": "test_001",
        "prompt": "寫一段程式碼。",
        "baseline_response": "print('hello')",
        "finetuned_response": "print('hello, world')",
        "target_response": "print('hello, world')"
    }]

    pipeline = EvaluationPipeline(
        provider_type="gemini",
        model="gemini-2.5-flash",
        max_concurrent=1,
        timeout=10.0,
        api_key="mock_key",
        api_url="[https://mock-endpoint.com](https://mock-endpoint.com)"
    )

    # Mock 裁判模型的非同步回傳值
    mock_judge_response_base = {"score": 3, "reason": "基本符合預期"}
    mock_judge_response_ft = {"score": 5, "reason": "完美對齊標答"}

    with patch.object(pipeline.judge, 'evaluate_single_case', SoftMockAsyncJudge(mock_judge_response_base, mock_judge_response_ft)):
        results = await pipeline.execute(mock_dataset)

        assert len(results) == 1
        res = results[0]
        assert res["id"] == "test_001"
        # 驗證統計相似度指標是否有獨立分離計算
        assert "base_metrics" in res
        assert "ft_metrics" in res
        assert res["ft_metrics"]["rouge1"] == 100.0  # Fine-tuned 完美對齊 Target
        assert res["judge_baseline_score"] == 3
        assert res["judge_finetuned_score"] == 5


@pytest.mark.asyncio
async def test_pipeline_fallback_flow_missing_ground_truth():
    """
    2. 驗證 Fallback 降級防禦：
       當資料集缺失 target_response 或為空字串時，
       管線不得崩潰，必須自動降級為原本的「雙模型自我對抗」模式。
       🌟 修正對齊防禦：為貼合 Token-level 物理重合度，Mock 兩端皆補入共同特徵詞 'SQL防禦'
    """
    mock_dataset = [{
        "id": "test_002",
        "prompt": "如何進行 SQL 防禦？",
        "baseline_response": "實施 SQL防禦 需要過濾單引號",
        "finetuned_response": "實施 SQL防禦 需要採用參數化查詢",
        "target_response": ""  # 🌟 故意缺失標準答案
    }]

    pipeline = EvaluationPipeline("gemini", "gemini-2.5-flash", 1, 10.0, "mock_key", "url")

    mock_judge_res = {"score": 3, "reason": "Fallback 盲測理由"}
    with patch.object(pipeline.judge, 'evaluate_single_case', AsyncMock(return_value=mock_judge_res)):
        results = await pipeline.execute(mock_dataset)

        res = results[0]
        # 驗證 Fallback 降級是否成功觸發，此時 base_metrics 必須徹底歸零
        assert res["base_metrics"]["rouge1"] == 0.0
        # 驗證自我對抗相似度：因為具備共同特徵詞，對抗分數絕對大於 0.0
        assert res["ft_metrics"]["rouge1"] > 0.0
        assert res["judge_baseline_score"] == 3


@pytest.mark.asyncio
async def test_pipeline_api_critical_failure_tolerance():
    """
    3. 驗證遠端 API 崩潰容錯性：
       當裁判核心 evaluate_single_case 因為外部網路或限流耗盡而丟回安全降級分時，
       Pipeline 主管道必須展現強大的容錯性，將 1 分與錯誤理由優雅落地，絕對不中斷死鎖。
    """
    mock_dataset = [{"id": "test_003", "prompt": "A", "baseline_response": "B", "finetuned_response": "C", "target_response": "A"}]
    pipeline = EvaluationPipeline("gemini", "gemini-2.5-flash", 1, 10.0, "mock_key", "url")

    # 模擬外部 API 終極失敗觸發的裁判降級資料
    mock_fallback_fail = {"score": 1, "reason": "API 呼叫終極失敗: 429 Too Many Requests"}
    with patch.object(pipeline.judge, 'evaluate_single_case', AsyncMock(return_value=mock_fallback_fail)):
        results = await pipeline.execute(mock_dataset)

        assert len(results) == 1
        assert results[0]["judge_baseline_score"] == 1
        assert "429 Too Many Requests" in results[0]["judge_baseline_reason"]


class SoftMockAsyncJudge:
    """輔助類：用於模擬序列化 Await 執行時，依序回傳 Baseline 與 Fine-tuned 的打分對象"""
    def __init__(self, base_return, ft_return):
        self.returns = [base_return, ft_return]
        self.call_count = 0

    async def __call__(self, *args, **kwargs):
        ret = self.returns[self.call_count]
        self.call_count += 1
        return ret