import os
import re
import sys
import subprocess
import tempfile
from pathlib import Path

# 需要批量处理的目录列表（相对或绝对路径均可）
DATASET_DIRS = [
    # 待打分目录列表
    "../model_answer/SLAM-Omni/hlt-lab_voicebench_commoneval_test",
    "../model_answer/SLAM-Omni/Jiann_STORAL_default_storal_en_test_short",
    "../model_answer/SLAM-Omni/truthfulqa_truthful_qa_generation_validation",
    "../model_answer/Tini-Omni/hlt-lab_voicebench_alpacaeval_test",
    "../model_answer/Tini-Omni/hlt-lab_voicebench_commoneval_test",
    "../model_answer/Tini-Omni/Jiann_STORAL_default_storal_en_test_short",
    "../model_answer/Tini-Omni/truthfulqa_truthful_qa_generation_validation"
]

# 源脚本路径
REPO_ROOT = Path("/root/zjy/SLAM-Omni")
WER_SRC = REPO_ROOT / "./wer.py"
GPT_SRC = REPO_ROOT / "./gpt_score.py"
UTMOS_SRC = REPO_ROOT / "./utmos.py"

PYTHON_BIN = sys.executable  # 使用当前解释器

# 简单的常量替换：替换脚本内 INPUT_PATH / OUTPUT_PATH 的赋值
INPUT_PATTERN = re.compile(r"^INPUT_PATH\s*=\s*\".*\"", re.M)
OUTPUT_PATTERN = re.compile(r"^OUTPUT_PATH\s*=\s*\".*\"", re.M)


def patch_and_run(script_path: Path, input_path: str, output_path: str):
    """在临时副本中替换路径常量并运行脚本（实时输出）。"""
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()
    code = INPUT_PATTERN.sub(f"INPUT_PATH = \"{input_path}\"", code)
    code = OUTPUT_PATTERN.sub(f"OUTPUT_PATH = \"{output_path}\"", code)

    tmp_dir = tempfile.mkdtemp(prefix="batch_score_")
    tmp_script = Path(tmp_dir) / script_path.name
    with open(tmp_script, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"[RUN] {tmp_script} -> input={input_path} output={output_path}")
    proc = subprocess.Popen([PYTHON_BIN, str(tmp_script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line, end="")
    except KeyboardInterrupt:
        proc.terminate()
        raise
    finally:
        ret = proc.wait()
    if ret != 0:
        print(f"[ERROR] {script_path.name} failed with exit code {ret}")
        raise SystemExit(ret)


def main():
    if not DATASET_DIRS:
        print("[ERROR] DATASET_DIRS 为空，请在脚本顶部填入7个需处理的目录路径。")
        sys.exit(1)

    for ds in DATASET_DIRS:
        ds_path = Path(ds)
        if not ds_path.exists():
            print(f"[WARN] 路径不存在，跳过: {ds}")
            continue
        manifest = ds_path / "manifest.jsonl"
        manifest_scored = ds_path / "manifest_scored.jsonl"
        # 1) WER：输入 manifest，输出 manifest_scored
        patch_and_run(WER_SRC, str(manifest), str(manifest_scored))
        # 2) GPT 评分：输入 manifest_scored，输出 manifest_scored（就地更新）
        patch_and_run(GPT_SRC, str(manifest_scored), str(manifest_scored))
        # 3) UTMOS：输入 manifest_scored，输出 manifest_scored（就地更新）
        patch_and_run(UTMOS_SRC, str(manifest_scored), str(manifest_scored))
        print(f"[DONE] {ds}\n")


if __name__ == "__main__":
    main()
