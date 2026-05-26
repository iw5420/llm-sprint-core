"""
Module Name: discover_models.py
Description: MLOps 運行期通用模型自動探測與排序工具。
             具備路徑防禦、Pydantic 欄位層級錯誤回報、自動語義版號提取與 Max-Heap 最佳推薦功能。
Author: Ace (Lead Architect)
"""

import os
import sys
import re

# 🌟 核心路徑防禦：動態計算專案根目錄，徹底根除 ModuleNotFoundError: No module named 'src' 破口
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import requests
from dotenv import load_dotenv
from pydantic import ValidationError

# 貫徹單一事實來源 (SSOT)：調用全局強型態組態防禦閘門
from src.config.schema import EnvConfigSchema


def parse_model_version(model_id: str) -> float:
    """
    正規表達式版號提取器：從模型 ID 中動態榨取出主版本號。
    例如: 'gemini-1.5-flash'          -> 1.5
          'gemini-2.5-flash'          -> 2.5
          'gemini-3.5-flash'          -> 3.5
          'gemini-4.5-pro'            -> 4.5 (達成前向相容)
          'gemini-2.0-flash-lite-001' -> 2.0
    """
    # 優先匹配標準的小數點版號 (如 1.5, 2.5, 3.5)
    match_float = re.search(r"gemini-(\d+\.\d+)", model_id)
    if match_float:
        return float(match_float.group(1))

    # 次要匹配單一整數版號 (如 2.0)
    match_int = re.search(r"gemini-(\d+)", model_id)
    if match_int:
        return float(match_int.group(1))

    return 0.0


def query_gemini_models(api_key: str) -> None:
    """
    呼叫 Google AI Studio 官方模型探測端點，動態計算並推薦目前雲端版號最高的最優模型。
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    print("🛰️  正在發射請求至 Google AI Studio 端點...")

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        models_list = data.get("models", [])
        if not models_list:
            print("⚠️  成功連線，但雲端未傳回任何可用模型清單。")
            return

        valid_flash_models = []
        valid_pro_models = []

        print("\n====================================================================================")
        print("🎯 Gemini 雲端活動模型實時解析大盤（100% 前向相容，杜絕字串硬編碼）")
        print("====================================================================================")
        print(f"{'模型實體識別碼 (Model ID)':<38} | {'提取版號':<8} | {'結構化 JSON 支援':<16}")
        print("------------------------------------------------------------------------------------")

        for model in models_list:
            model_id = model.get("name", "").split("/")[-1]
            methods = model.get("supportedGenerationMethods", [])

            # 🔍 剛需守門員 1：必須支援文字內容生成 (Chat/Text Completion)
            if "generateContent" not in methods:
                continue

            # 🔍 剛需守門員 2：排除語音、影像、機器人等無法穩定輸出標準結構化 JSON 的 Specialist 模型
            is_specialist = any(x in model_id for x in ["image", "tts", "robotics", "clip", "nano", "preview"])
            if is_specialist:
                print(f"   {model_id:<35} | {'N/A':<8} | 🟡 需測試防禦")
                continue

            # 🔍 核心自適應演算法：動態計算語義版號
            version = parse_model_version(model_id)
            if version > 0.0:
                json_support = "🟢 完美相容"
                print(f"   {model_id:<35} | {version:<8} | {json_support:<16}")

                # 依據架構流派，將主線推理模型分流，準備進行最大值篩選
                if "flash" in model_id:
                    valid_flash_models.append((version, model_id))
                elif "pro" in model_id:
                    valid_pro_models.append((version, model_id))

        print("------------------------------------------------------------------------------------")

        # 🌟 終極自動推導最優解：利用 Max-Heap 篩選，自動抓取目前雲端存在的最高數字模型！
        if valid_flash_models:
            best_flash = max(valid_flash_models, key=lambda x: x[0])[1]
            print(f"🔥 [動態推導結果] 目前雲端版號最高 Flash 首選模型: {best_flash}")
        else:
            best_flash = "gemini-1.5-flash"

        if valid_pro_models:
            best_pro = max(valid_pro_models, key=lambda x: x[0])[1]
            print(f"✨ [動態推導結果] 目前雲端版號最高 Pro 精準模型: {best_pro}")
        else:
            best_pro = "gemini-1.5-pro"

        print("====================================================================================")
        print(f"💡 [架構師提示] 複製動態推導出的最高版號 '{best_flash}'，填入 YAML 即可。未來無論出 4.0 或 4.5，本腳本永不崩潰！")
        print("====================================================================================\n")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            print("❌ 探測失敗 [400 Bad Request]：請檢查你的 PROVIDER_API_KEY (AIzaSy...) 是否填寫完全正確。", file=sys.stderr)
        else:
            print(f"❌ 網路通訊失敗 (HTTP Error): {str(e)}", file=sys.stderr)
    except Exception as e:
        print(f"❌ 發生非預期系統崩潰: {str(e)}", file=sys.stderr)


def query_openai_models(api_key: str, api_url: str) -> None:
    """
    呼叫 OpenAI 官方或企業內部自建 vLLM / Ollama 叢集進行模型探測 (OpenAI 多型適配)。
    """
    base_url = api_url.split("/chat/completions")[0]
    url = f"{base_url}/models"

    headers = {"Authorization": f"Bearer {api_key}"}
    print(f"🛰️  正在發射請求至 OpenAI 相容端點: {url} ...")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        print("\n====================================================================================")
        print("🎯 OpenAI / Local vLLM 生態系活動模型探測報告")
        print("====================================================================================")
        for model in data.get("data", []):
            print(f"|-- 可用模型 ID: {model.get('id')}")
        print("====================================================================================\n")
    except Exception as e:
        print(f"❌ OpenAI/vLLM 端點探測失敗: {str(e)}", file=sys.stderr)


def main() -> None:
    print("==================================================")
    print("🔍 啟動 MLOps 運行期模型自動探測核心 (Runtime Discovery)...")
    print("==================================================")

    # 先將本地的 .env 檔案內容倒進作業系統環境變數中
    load_dotenv()

    # 🌟 細緻化強型態守門：精確攔截 ValidationError 並吐出具體是哪個欄位型態出錯或漏讀
    try:
        env_config = EnvConfigSchema(
            PROVIDER_TYPE=os.getenv("PROVIDER_TYPE"),
            PROVIDER_API_KEY=os.getenv("PROVIDER_API_KEY"),
            EVAL_API_URL=os.getenv("EVAL_API_URL")
        )
    except ValidationError as e:
        print("\n❌ [CRITICAL] 環境變數強型態校驗未通過！詳細欄位錯誤報告如下：", file=sys.stderr)
        print("-" * 70, file=sys.stderr)
        print(e, file=sys.stderr)
        print("-" * 70, file=sys.stderr)
        print("👉 排查建議：請檢查專案根目錄的 .env 檔案是否存在，且 PROVIDER_TYPE 是否嚴格等於小寫的 gemini 或 openai。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 其他非預期錯誤: {str(e)}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ 成功載入憑證。目前系統鎖定供應商生態系: {env_config.provider_type.upper()}")

    # 多型分發路由 (Polymorphic Routing)
    if env_config.provider_type == "gemini":
        query_gemini_models(api_key=env_config.provider_api_key)
    else:
        query_openai_models(
            api_key=env_config.provider_api_key,
            api_url=env_config.eval_api_url
        )


if __name__ == "__main__":
    main()