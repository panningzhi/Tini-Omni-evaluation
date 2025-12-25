#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_pred_text_to_manifest.py

Merge predicted text and predicted audio paths into a new manifest.jsonl.
Jiann_STORAL_default_storal_en_test
truthfulqa_truthful_qa_generation_validation
hlt-lab_voicebench_commoneval_test
hlt-lab_voicebench_alpacaeval_test

Usage example:

    python /root/pnz/SLAM-Omni/examples/s2s/scripts/evaluation/merge_pred_text_to_manifest.py \
        --base_path /root/autodl-tmp \
        --ckpt_name model \
        --test_data_name hlt-lab_voicebench_alpacaeval_test \
        --voice_prompt_id prompt-6

"""


import os
import json
import argparse

def load_pred_text_map(pred_text_path):
    """
    pred_text 格式：
    1.wav this is predicted text...
    2.wav another text...

    返回 dict: {1: "this is predicted text...", ...}
    """
    id2pred = {}
    with open(pred_text_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            if not line:
                continue

            # 按第一个空白分隔（支持多空格或tab）
            parts = line.split(None, 1)

            if len(parts) != 2:
                print(f"[WARN] Unrecognized line, skipped: {line}")
                continue

            key, text = parts
            if not key.endswith(".wav"):
                print(f"[WARN] Skip line without .wav key: {line}")
                continue

            # 去掉 .wav 得到 ID
            try:
                row_id = int(key.replace(".wav", ""))
            except:
                print(f"[WARN] Bad key format: {key}")
                continue
            id2pred[row_id] = text.strip()

    return id2pred


def main(base_path, ckpt_name, test_data_name, voice_prompt_id):
    # ---------------- 路径 ----------------
    old_manifest = f"{base_path}/evaluation/voice_prompt/{test_data_name}/manifest.jsonl"
    pred_text_path = f"{base_path}/evaluation/model_answer/{ckpt_name}/{test_data_name}/pred_text"
    new_manifest = f"{base_path}/evaluation/model_answer/{ckpt_name}/{test_data_name}/manifest.jsonl"

    os.makedirs(os.path.dirname(new_manifest), exist_ok=True)

    # ---------------- 加载预测文本映射 ----------------
    id2pred = load_pred_text_map(pred_text_path)
    print(f"[INFO] Loaded pred_text entries: {len(id2pred)}")

    # ---------------- 读取旧 manifest.jsonl ----------------
    rows = []
    with open(old_manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    print(f"[INFO] Loaded manifest rows: {len(rows)}")

    # ---------------- 处理并生成 new manifest ----------------
    with open(new_manifest, "w", encoding="utf-8") as fout:
        for row in rows:
            new_row = dict(row)  # 不修改原 row

            row_id = row["id"]
            pred_text = id2pred.get(row_id, "")

            new_row["generated_text"] = pred_text

            # 更新 wav_path
            new_row["wav_path"] = (
                f"{base_path}/evaluation/model_answer/"
                f"{ckpt_name}/{test_data_name}/pred_audio/"
                f"{voice_prompt_id}/{row_id}.wav"
            )

            fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")

    print(f"[DONE] Written to: {new_manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/root/autodl-tmp/evaluation")
    parser.add_argument("--ckpt_name", type=str, required=True)
    parser.add_argument("--test_data_name", type=str, required=True)
    parser.add_argument("--voice_prompt_id", type=str, required=True)

    args = parser.parse_args()

    main(args.base_path, args.ckpt_name, args.test_data_name, args.voice_prompt_id)
