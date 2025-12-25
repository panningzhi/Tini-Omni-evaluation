"""批量计算语音转写与文本的 WER 并写入新 JSONL

读取固定输入文件 manifest.jsonl，每行 JSON 必须包含：
  - wav_path: 音频文件路径（生成的答案的语音）
  - generated_text: 该语音对应的文本（作为参考文本）

输出：在原字段基础上追加键 "wer" 写入 manifest_scored.jsonl。
如果转写或文件缺失则 wer 置为 null。
"""

import os
import json
import torch
import whisper
from jiwer import wer

INPUT_PATH = "/root/autodl-tmp/evaluation/model_answer/model/hlt-lab_voicebench_alpacaeval_test/manifest.jsonl"
OUTPUT_PATH = "/root/autodl-tmp/evaluation/model_answer/model/hlt-lab_voicebench_alpacaeval_test/manifest_scored.jsonl"
WHISPER_MODEL = "/root/autodl-tmp/whisper/small.pt"  # 可换成 "small" 等名称
LANGUAGE = "en"  # 若希望自动检测可设为 None

def iter_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line=line.strip()
            if not line:
                continue
            try:
                obj=json.loads(line)
            except Exception as e:
                print(f"[PARSE-ERROR] line={line_no}: {e}")
                continue
            yield line_no, obj

def main():
    print(f"[INFO] 输入清单: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 使用设备: {device}")
    model = whisper.load_model(WHISPER_MODEL, device=device)

    total=0
    done=0
    wers=[]
    results=[]

    # 正式遍历
    for line_no, rec in iter_jsonl(INPUT_PATH):
        total += 1
        wav_path = rec.get('wav_path')
        ref_text = rec.get('generated_text')  # 参考文本选择生成文本
        if not wav_path or not os.path.exists(wav_path):
            ap = os.path.abspath(wav_path) if wav_path else None
            print(f"[WARN] line={line_no} id={rec.get('id')} 缺少或找不到音频: raw='{wav_path}' abs='{ap}' exists={os.path.exists(ap) if ap else False}")
            rec['wer'] = None
            results.append(rec)
            continue
        if not isinstance(ref_text, str) or not ref_text.strip():
            print(f"[WARN] line={line_no} id={rec.get('id')} 参考文本为空")
            rec['wer'] = None
            results.append(rec)
            continue
        # 转写
        try:
            if LANGUAGE:
                tr = model.transcribe(wav_path, language=LANGUAGE)
            else:
                tr = model.transcribe(wav_path)
            hyp = (tr.get('text') or '').strip()
        except Exception as e:
            print(f"[ERROR] line={line_no} id={rec.get('id')} 转写失败: {e}")
            rec['wer'] = None
            results.append(rec)
            continue
        # WER
        try:
            score = wer(ref_text.strip(), hyp)
            rec['wer'] = score
            wers.append(score)
            done += 1
            print(f"[OK] id={rec.get('id')} WER={score:.4f}")
        except Exception as e:
            print(f"[ERROR] line={line_no} id={rec.get('id')} 计算 WER 失败: {e}")
            rec['wer'] = None
        results.append(rec)

    if wers:
        print(f"\n[SUMMARY] 成功 {done}/{total} 平均 WER={sum(wers)/len(wers):.4f}")
    else:
        print(f"\n[SUMMARY] 无成功样本。total={total}")

    out_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"[INFO] 写入完成: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()