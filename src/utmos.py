import json
import os
import tempfile
import shutil
import utmosv2

# ===== 路径自己改 =====
INPUT_PATH = "../model_answer/SLAM-Omni/hlt-lab_voicebench_alpacaeval_test/manifest_scored.jsonl"
OUTPUT_PATH = "../model_answer/SLAM-Omni/hlt-lab_voicebench_alpacaeval_test/manifest_scored.jsonl"

# 创建 UTMOS 模型（用 GPU）
model = utmosv2.create_model(pretrained=True, device="cuda")

def get_mos_for_wav(wav_path: str):
    """
    给一个 wav 路径，调用 utmosv2 返回 MOS 分数。
    这里假设 model.predict(input_path=...) 返回一个 float 或长度为 1 的 list。
    如果你之前跑过可以根据实际再微调。
    """
    mos = model.predict(input_path=wav_path)

    # 兼容几种可能的返回类型
    if isinstance(mos, (list, tuple)):
        mos = mos[0]
    elif isinstance(mos, dict):
        # 如果是 {path: score}
        mos = list(mos.values())[0]

    return float(mos)

def _process_stream(fin, fout):
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue

        wav_path = item.get("wav_path")
        if not wav_path or not os.path.exists(wav_path):
            item["utmos_mos"] = None
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"id={item.get('id')}  wav={wav_path}  MOS=None (missing)")
            continue

        try:
            mos_score = get_mos_for_wav(wav_path)
        except Exception as e:
            item["utmos_mos"] = None
            item["error_utmos"] = f"{type(e).__name__}: {str(e)[:200]}"
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"id={item.get('id')}  wav={wav_path}  MOS=None (error)")
            continue

        item["utmos_mos"] = mos_score
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"id={item.get('id')}  wav={wav_path}  MOS={mos_score:.3f}")


def main():
    same_path = os.path.abspath(INPUT_PATH) == os.path.abspath(OUTPUT_PATH)

    if same_path:
        # 安全的就地更新：写到同目录临时文件，完成后原子替换
        out_dir = os.path.dirname(os.path.abspath(OUTPUT_PATH)) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="utmos_tmp_", suffix=".jsonl", dir=out_dir)
        os.close(fd)
        try:
            with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
                 open(tmp_path, "w", encoding="utf-8") as fout:
                _process_stream(fin, fout)
            # 原子替换，避免中途失败导致文件丢失
            os.replace(tmp_path, OUTPUT_PATH)
        finally:
            # 若发生异常，清理临时文件
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    else:
        # 正常地读一个文件，写到另一个文件
        with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
             open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
            _process_stream(fin, fout)

if __name__ == "__main__":
    main()