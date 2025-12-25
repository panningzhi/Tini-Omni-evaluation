#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tts_from_test_jsonl.py

Batch TTS from text_prompt JSONL -> generate wav + manifest.jsonl

Features:
  - Only enforce max_duration_s (default: 30s)
  - If generated audio is longer → TRUNCATE
  - If shorter → DO NOTHING (NO padding)

Usage:
  python /root/pnz/SLAM-Omni/examples/s2s/scripts/evaluation/tts_from_test_jsonl.py truthfulqa_truthful_qa_generation_validation --max_duration_s 30
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# ------------------ 固定路径 ------------------
BASE_PATH = "/root/autodl-tmp/evaluation"
TEXT_PROMPT_DIR = f"{BASE_PATH}/text_prompt"
VOICE_PROMPT_DIR = f"{BASE_PATH}/voice_prompt"

DEFAULT_COSYVOICE_PATH = "/root/autodl-tmp/codec/CosyVoice-300M-SFT"
SAMPLE_RATE = 22050
DEFAULT_SPK = "英文女"

# ------------------ 测试控制：最多处理多少条（None = 全部） ------------------
MAX_ROWS = 200  # 例如：设为 10 表示只跑前 10 条；默认 None 全部处理


# ------------------ 你提供的可运行 CosyVoice 导入写法 ------------------
MATCHA_PATH = "/root/pnz/SLAM-Omni/examples/s2s/utils/third_party/Matcha-TTS"
if MATCHA_PATH not in sys.path:
    sys.path.append(MATCHA_PATH)

COSY_ROOT = "/root/pnz/SLAM-Omni/examples/s2s/utils"
if COSY_ROOT not in sys.path:
    sys.path.insert(0, COSY_ROOT)

try:
    from cosyvoice.cli.cosyvoice import CosyVoice
except Exception as e:
    print("ERROR: fail to import cosyvoice", e)
    raise

import torch
import soundfile as sf

# ------------------ 字段映射 ------------------
DATASET_FIELD_MAP = {
    "Jiann_STORAL_default_storal_en_test": {"source_key": "story", "target_key": "moral"},
    "truthfulqa_truthful_qa_generation_validation": {"source_key": "question", "target_key": "best_answer"},
    "hlt-lab_voicebench_commoneval_test": {"source_key": "question", "target_key": "best_answer"},
}

# ------------------ CosyVoice 调用 ------------------
def run_cosyvoice(cosy, text: str, spk_id: str, stream: bool = False):
    outputs = []
    for out in cosy.inference_sft(text, spk_id, stream=stream):
        if "tts_speech" in out and out["tts_speech"] is not None:
            outputs.append(out["tts_speech"])

    if len(outputs) == 0:
        raise RuntimeError("CosyVoice生成空输出")

    total = torch.cat(outputs, dim=-1)
    audio = total.cpu().numpy().flatten().astype("float32")
    return audio

# ------------------ 截断逻辑（唯一保留） ------------------
def truncate_audio(audio_np, sample_rate, max_duration_s):
    max_samples = int(max_duration_s * sample_rate)
    if audio_np.shape[0] > max_samples:
        audio_np = audio_np[:max_samples]
    return audio_np


# ------------------ 主流程 ------------------
def run_tts(test_name: str, cosyvoice_path: str, max_duration_s: float = 30.0):
    input_jsonl = f"{TEXT_PROMPT_DIR}/{test_name}.jsonl"
    if not os.path.exists(input_jsonl):
        raise FileNotFoundError(input_jsonl)

    output_dir = f"{VOICE_PROMPT_DIR}/{test_name}"
    os.makedirs(output_dir, exist_ok=True)

    # load cosy
    cosy = CosyVoice(cosyvoice_path, load_jit=True, load_onnx=False, fp16=True)

    # field mapping
    field_map = DATASET_FIELD_MAP.get(test_name)
    if field_map is None:
        raise RuntimeError(f"未配置字段映射: {test_name}")

    source_key = field_map["source_key"]
    target_key = field_map["target_key"]

    manifest_path = f"{output_dir}/manifest.jsonl"
    fout = open(manifest_path, "w", encoding="utf-8")

    # 读取文本
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            # --- 控制最大行数 ---
            if MAX_ROWS is not None and idx > MAX_ROWS:
                print(f"[INFO] 已达到 MAX_ROWS={MAX_ROWS}，停止提前退出。")
                break
            
            item = json.loads(line)
            text = item[source_key]
            target_text = item.get(target_key) if target_key else None
            wav_path = f"{output_dir}/{idx}.wav"
            # 打印进度
            print(f"[{idx+1}/{MAX_ROWS}] Synthesizing text length {len(text)}")
            # --- generate ---
            audio_np = run_cosyvoice(cosy, text, spk_id=DEFAULT_SPK)

            # --- truncate ---
            audio_np = truncate_audio(audio_np, SAMPLE_RATE, max_duration_s)

            # write wav
            sf.write(wav_path, audio_np, SAMPLE_RATE)

            # duration
            duration = len(audio_np) / SAMPLE_RATE

            # manifest
            fout.write(json.dumps({
                "id": idx,
                "key": os.path.basename(wav_path), 
                "source_wav": wav_path,
                "source_text": text,
                "target_text": target_text,
                "duration": duration,
            }, ensure_ascii=False) + "\n")

    fout.close()
    print(f"[DONE] Saved to {output_dir}")


# ------------------ CLI ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_name", type=str)
    parser.add_argument("--cosyvoice_path", type=str, default=DEFAULT_COSYVOICE_PATH)
    parser.add_argument("--max_duration_s", type=float, default=30.0)

    args = parser.parse_args()

    run_tts(args.test_name, args.cosyvoice_path, args.max_duration_s)
