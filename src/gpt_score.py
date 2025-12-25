import os
import json
import time
import tempfile
import random
import re
from typing import List, Dict
import requests

# ===== 文件路径（注意：我这里用绝对路径示例，你按实际路径改） =====
INPUT_PATH = "../model_answer/model/hlt-lab_voicebench_alpacaeval_test/manifest_scored.jsonl"
OUTPUT_PATH = "../model_answer/model/hlt-lab_voicebench_alpacaeval_test/manifest_scored.jsonl"
# INPUT_PATH = "../model_answer/model/Jiann_STORAL_default_storal_en_test/mani_test.jsonl"
# OUTPUT_PATH = "../model_answer/model/Jiann_STORAL_default_storal_en_test/mani_test.jsonl"
# ===== 打分提示词 =====
SYSTEM_PROMPT = """I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output. Your task is to rate the model’s responses based on the provided user input transcription [Instruction], the model’s output transcription [Response] and some suggested answers [Reference]. The model’s response doesn’t necessarily have to be identical to the suggested answers, as long as it aligns with the question and is reasonable.

Please evaluate the response on a scale of 1 to 5:

1 point: The response is largely irrelevant, incorrect, or fails to address the user’s query. It may be off-topic or provide incorrect information. The response does not align with the question in any meaningful way.
2 points: The response is somewhat relevant but lacks accuracy, completeness, or coherence. It may partially address the query but introduces unnecessary information or deviates from the core issue. The response may not align well with the suggested answer but still provides some value.
3 points: The response is relevant and mostly accurate, but may lack conciseness or clarity. It addresses the question reasonably, but there might be slight deviations in approach or content. While it may not strictly align with the suggested answer, it still effectively addresses the core of the query.
4 points: The response is relevant, accurate, and concise. It provides a clear answer to the user’s question and avoids unnecessary details. While it may not exactly mirror the suggested answer, it effectively addresses the user’s query in a logical and well-reasoned manner.
5 points: The response is exceptionally relevant, accurate, and concise. It directly addresses the user’s query in the most efficient manner, providing exactly the information needed. The response may differ from the suggested answer in phrasing or approach but still aligns perfectly with the intent of the query, demonstrating a high level of reasoning and clarity.
"""

USER_PROMPT_TEMPLATE = """Below are the transcription of user’s instruction, models’ response and the reference answer:
### [Instruction] {question}
### [Response] {answer}
### [Reference] {reference}
After evaluating, please output the score only without anything else. You don’t need to provide any explanations."""

# ===== newapi 相关配置 =====
DEFAULT_JUDGE_MODEL = os.environ.get("GPT_JUDGE_MODEL", "gpt-5-chat")
API_KEY_ENV = "NEWAPI_API_KEY"
API_BASE_URL = os.environ.get("NEWAPI_BASE_URL", "https://api.newapi.com/v1")


def _call_newapi(model: str, messages: List[Dict], temperature: float = 0.1, max_retries: int = 5) -> str:
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise ValueError(f"Please set {API_KEY_ENV} environment variable")

    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    backoff = 1.0
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            # Retry on 429 or 5xx
            if response.status_code in (429, 500, 502, 503, 504):
                last_err = Exception(f"HTTP {response.status_code}: {response.text[:200]}")
                raise last_err
            response.raise_for_status()
            result = response.json()
            # Some gateways may return content in different structure; try best-effort extraction
            content = None
            try:
                content = result["choices"][0]["message"]["content"]
            except Exception:
                # Fallbacks for alternative formats
                content = (
                    result.get("choices", [{}])[0].get("text")
                    or result.get("message", {}).get("content")
                    or result.get("output")
                )
            if not content:
                raise ValueError(f"Unexpected API response format: {result}")
            return str(content).strip()
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                raise last_err
            sleep_s = backoff + random.uniform(0, 0.5)
            print(f"[WARN] API call failed (attempt {attempt}/{max_retries}): {e}. Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
            backoff = min(backoff * 2, 16)


# ===== 打分逻辑 =====
def score_one(question: str, pred: str, ref: str):
    """对一条 (question, pred, ref) 进行打分"""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question, answer=pred, reference=ref
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    text = _call_newapi(DEFAULT_JUDGE_MODEL, messages, temperature=0.0)

    # 解析仅数字分数，回退正则
    score = None
    try:
        # 先尝试纯数字
        score = float(text.strip())
    except Exception:
        try:
            nums = re.findall(r"\d+(?:\.\d+)?", text)
            score = float(nums[0]) if nums else None
        except Exception:
            print(f"[WARN] 解析分数失败，返回原文：{text}")
            score = None
    if score is not None:
        score = max(0.0, min(5.0, score))
    return score, text


def _process_stream(fin, fout):
    for line in fin:
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)

        qid = item.get("id")
        question = item.get("source_text", "")
        ref = item.get("target_text", "")
        pred = item.get("generated_text", "")

        try:
            score, raw_text = score_one(question, pred, ref)
            item["chatgpt_score"] = score
            item["raw_model_output"] = raw_text
        except Exception as e:
            item["chatgpt_score"] = None
            item["raw_model_output"] = None
            item["error"] = f"{type(e).__name__}: {str(e)[:200]}"

        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"id={qid} score={item.get('chatgpt_score')} err={item.get('error')}")
        time.sleep(0.2)


def main():
    same_path = os.path.abspath(INPUT_PATH) == os.path.abspath(OUTPUT_PATH)

    if same_path:
        out_dir = os.path.dirname(os.path.abspath(OUTPUT_PATH)) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="gptscore_tmp_", suffix=".jsonl", dir=out_dir)
        os.close(fd)
        try:
            with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
                 open(tmp_path, "w", encoding="utf-8") as fout:
                _process_stream(fin, fout)
            os.replace(tmp_path, OUTPUT_PATH)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    else:
        with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
             open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
            _process_stream(fin, fout)


if __name__ == "__main__":
    main()