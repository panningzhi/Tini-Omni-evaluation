#!/usr/bin/env python
# avg_metrics_fixed_path.py

import json

# 支持批量展示多个数据集的 manifest_scored.jsonl
INPUT_PATHS = [
    # 模型 SLAM-Omni 根下
    "../model_answer/SLAM-Omni/hlt-lab_voicebench_alpacaeval_test/manifest_scored.jsonl",
    "../model_answer/SLAM-Omni/hlt-lab_voicebench_commoneval_test/manifest_scored.jsonl",
    "../model_answer/SLAM-Omni/Jiann_STORAL_default_storal_en_test_short/manifest_scored.jsonl",
    "../model_answer/SLAM-Omni/truthfulqa_truthful_qa_generation_validation/manifest_scored.jsonl",
    # 模型 Tini-Omni 根下
    "../model_answer/Tini-Omni/hlt-lab_voicebench_alpacaeval_test/manifest_scored.jsonl",
    "../model_answer/Tini-Omni/hlt-lab_voicebench_commoneval_test/manifest_scored.jsonl",
    "../model_answer/Tini-Omni/Jiann_STORAL_default_storal_en_test_short/manifest_scored.jsonl",
    "../model_answer/Tini-Omni/truthfulqa_truthful_qa_generation_validation/manifest_scored.jsonl",
]

# 根据你实际字段名改
WER_KEY = "wer"
GPT_KEY = "chatgpt_score"
UTMOS_KEY = "utmos_mos"


def summarize_one(path: str):
    wer_sum = 0.0
    wer_cnt = 0
    gpt_sum = 0.0
    gpt_cnt = 0
    utmos_sum = 0.0
    utmos_cnt = 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] {path} line {line_no} 不是合法 JSON，跳过")
                    continue

                # WER
                v = item.get(WER_KEY)
                if v is not None:
                    try:
                        wer_sum += float(v)
                        wer_cnt += 1
                    except (TypeError, ValueError):
                        print(f"[WARN] {path} line {line_no} {WER_KEY} 不是数值，跳过")

                # ChatGPT score
                v = item.get(GPT_KEY)
                if v is not None:
                    try:
                        gpt_sum += float(v)
                        gpt_cnt += 1
                    except (TypeError, ValueError):
                        print(f"[WARN] {path} line {line_no} {GPT_KEY} 不是数值，跳过")

                # UTMOS
                v = item.get(UTMOS_KEY)
                if v is not None:
                    try:
                        utmos_sum += float(v)
                        utmos_cnt += 1
                    except (TypeError, ValueError):
                        print(f"[WARN] {path} line {line_no} {UTMOS_KEY} 不是数值，跳过")
    except FileNotFoundError:
        print(f"[WARN] 文件不存在，跳过: {path}")
        return None

    def safe_avg(total, count):
        return total / count if count > 0 else None

    return {
        "wer_cnt": wer_cnt,
        "wer_avg": safe_avg(wer_sum, wer_cnt),
        "gpt_cnt": gpt_cnt,
        "gpt_avg": safe_avg(gpt_sum, gpt_cnt),
        "utmos_cnt": utmos_cnt,
        "utmos_avg": safe_avg(utmos_sum, utmos_cnt),
    }


def pretty_label(path: str):
    # 例：../model_answer/Tini-Omni/hlt-lab_voicebench_alpacaeval_test/manifest_scored.jsonl
    parts = path.strip("/").split("/")
    # 模型名: 紧邻 model_answer 之后的目录名（如 model 或 Tini-Omni）
    try:
        i = parts.index("model_answer")
        model_name = parts[i+1]
    except Exception:
        model_name = "unknown_model"
    # 数据集名: 倒数第二个目录名
    dataset_name = parts[-2] if len(parts) >= 2 else "unknown_dataset"
    return model_name, dataset_name


def main():
    print("===== Metrics Overview =====")
    for p in INPUT_PATHS:
        model_name, dataset_name = pretty_label(p)
        stats = summarize_one(p)
        if not stats:
            continue
        print(f"\n[{model_name}] {dataset_name}")
        print(f"WER   count = {stats['wer_cnt']},   avg = {stats['wer_avg']}")
        print(f"GPT   count = {stats['gpt_cnt']},   avg = {stats['gpt_avg']}")
        print(f"UTMOS count = {stats['utmos_cnt']}, avg = {stats['utmos_avg']}")


if __name__ == "__main__":
    main()