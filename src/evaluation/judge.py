"""
Module Name: judge.py
Description: 多廠商自適應非同步裁判核心 (完全體觀測版)
"""

import json
import asyncio
import sys
import aiohttp
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

class JudgeResponseSchema(BaseModel):
    score: int = Field(..., ge=1, le=5)
    reason: str = Field(..., min_length=5)

class AsyncLLMJudge:
    def __init__(self, provider_type: str, model: str, max_concurrent: int, timeout: float, api_key: str, api_url: str):
        self.provider_type = provider_type.lower()
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.api_key = api_key
        self.api_url = api_url

    def _prepare_request_meta(self) -> tuple[str, dict]:
        headers = {"Content-Type": "application/json"}
        # 全面走 2026 標準 Bearer Token 簽章
        headers["Authorization"] = f"Bearer {self.api_key}"
        return self.api_url, headers

    def _build_rubric_prompt(self, prompt: str, response: str) -> list:
        full_instruction = (
            "【系統裁判規則】\n"
            "你是一位極其嚴格的 AI 代碼與邏輯評估裁判。\n"
            "請針對使用者的提示詞與模型輸出的回答進行盲測打分，評分範圍 1 到 5 分：\n"
            "1分：完全離題或產生嚴重幻覺。\n"
            "3分：基本答對，但缺乏深度。\n"
            "5分：完美回答，邏輯嚴密。\n"
            "你必須嚴格以 JSON 格式回傳，不得包含任何 Markdown 標記。\n"
            "格式範例：{\"score\": 5, \"reason\": \"原因...\"}\n\n"
            "--------------------------------------------------\n"
            "【待評估盲測數據】\n"
            f"使用者提示詞: {prompt}\n\n"
            f"模型回答內容: {response}"
        )
        return [{"role": "user", "content": full_instruction}]

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_random_exponential(min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _call_api_with_retry(self, session: aiohttp.ClientSession, payload: dict) -> str:
        target_url, headers = self._prepare_request_meta()
        async with session.post(target_url, headers=headers, json=payload, timeout=self.timeout) as response:
            if response.status == 429:
                print("⚠️  [Rate Limit] 觸發 429 限流，看門狗正在調度非同步指數退避重試...")
                raise aiohttp.ClientResponseError(response.request_info, response.history, status=429)

            if response.status != 200:
                err_text = await response.text()
                print(f"❌ [HTTP ERROR] 狀態碼: {response.status}, 內容: {err_text}", file=sys.stderr)
                response.raise_for_status()

            res_json = await response.json()

            # 自適應多型解析核心
            if "choices" in res_json and len(res_json["choices"]) > 0:
                return res_json["choices"][0]["message"]["content"]
            if "candidates" in res_json and len(res_json["candidates"]) > 0:
                return res_json["candidates"][0]["content"]["parts"][0].get("text", "")

            raise KeyError(f"無法從多廠商響應體中榨取出有效文本欄位。")

    async def evaluate_single_case(self, session: aiohttp.ClientSession, prompt: str, response: str) -> dict:
        async with self.semaphore:
            payload = {
                "model": self.model,
                "messages": self._build_rubric_prompt(prompt, response),
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
            try:
                raw_content = await self._call_api_with_retry(session, payload)

                if "```json" in raw_content:
                    raw_content = raw_content.split("```json")[-1].split("```")[0].strip()
                elif "```" in raw_content:
                    raw_content = raw_content.split("```")[-1].split("```")[0].strip()

                data = json.loads(raw_content)
                validated = JudgeResponseSchema.model_validate(data)

                # 🌟 實時觀測點：確認到底有沒有成功拿到正常分數
                print(f"🟢 [Judge 成功解鎖] 模型: {self.model} -> 實時打分: {validated.score} 分")
                return {"score": validated.score, "reason": validated.reason}

            except Exception as e:
                # 終極守護 (Fallback)
                print(f"🔴 [Judge 觸發降級] 錯誤原因: {str(e)}", file=sys.stderr)
                return {"score": 1, "reason": f"API 呼叫終極失敗: {str(e)}"}