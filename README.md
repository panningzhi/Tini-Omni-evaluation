# Tini-Omni-Evaluation

This repository serves as the evaluation framework for **[Tini-Omni](https://github.com/hypJean/Tini-Omni)** and is designed to be a versatile evaluation system for various Spoken Dialogue Models (SDMs).

## Overview

The goal of this framework is to provide an engineering-ready design for reproducing evaluation experiments. It covers the entire pipeline from data preparation to multi-dimensional scoring.

### Core Objectives
- **Sample Extraction**: Extract 200 samples from each HuggingFace test set.
- **TTS Synthesis**: Generate user instruction audio prompts using **CosyVoice-300M**.
- **Model Inference**: Evaluate SDMs by generating response audio and text.
- **Comprehensive Metrics**:
  - **ChatGPT Score**: Content quality assessment using ChatGPT.
  - **ASR-WER**: Speech-to-text accuracy using Whisper-small.
  - **UTMOS**: Speech naturalness and quality prediction.

## Directory Structure & Naming Convention

The project follows a strict directory structure to ensure consistency across different models and datasets.

```text
evaluation/
├─ text_prompt/                        # Step 1: Raw text samples (jsonl)
│  ├─ Jiann_STORAL_default_storal_en_test.jsonl
│  └─ ...
├─ voice_prompt/                       # Step 2: Synthesized source wav + manifest
│  ├─ Jiann_STORAL_default_storal_en_test/
│  │  ├─ 1.wav
│  │  ├─ 2.wav
│  │  └─ manifest.jsonl
├─ model_answer/                       # Step 3: Model outputs (Audio + Text)
│  ├─ SLAM-Omni/                       # Model Name / Checkpoint Name
│  │  ├─ Jiann_STORAL_default_storal_en_test/
│  │  │  ├─ pred_audio/                # Generated Audio Files
│  │  │  │  ├─ prompt-6/
│  │  │  │  │  ├─ 1.wav
│  │  │  │  │  └─ ...
│  │  │  ├─ question_text              # Input Questions
│  │  │  └─ pred_text                  # Model Predicted Text
│  │  └─ ...
│  └─ Tini-Omni/
│     └─ ...
├─ eval_results/                       # Step 4: Evaluation Scores
│  ├─ SLAM-Omni/
│  │  ├─ Jiann_STORAL_default_storal_en_test/
│  │  │  ├─ chatgpt_score.jsonl
│  │  │  ├─ asr_wer.jsonl
│  │  │  └─ utmos.jsonl
│  │  └─ ...
│  └─ ...
```

**Naming Convention**:
- File/Directory names must include the `test_data_name` derived from HuggingFace (e.g., `Jiann_STORAL_default_storal_en_test`).
- All JSONL files use `id` (1..200) as the primary key for each sample.

## Step-by-Step Implementation

### Step 1: Prepare Test Sets
**Script**: `src/download_test_json_from_huggingface.py`

Downloads 200 samples from specified HuggingFace datasets and saves them as JSONL files in `text_prompt/`.

- **Input**: HuggingFace Dataset ID (e.g., `Jiann/STORAL`)
- **Output**: `${base_path}/evaluation/text_prompt/${test_data_name}.jsonl`

### Step 2: Synthesize Voice Prompts (TTS)
**Script**: `src/tts_from_test_jsonl.py`

Synthesizes audio prompts for user instructions using CosyVoice-300M.

- **Input**: JSONL files from Step 1.
- **Output**: 
  - Audio files in `voice_prompt/${test_data_name}/`
  - `manifest.jsonl` mapping text to audio paths.
- **Configuration**: Maps dataset fields (e.g., `story`, `instruction`) to source text for TTS.

### Step 3: Batch Inference
**Script**: `inference_s2s_batch.sh` (User Implementation Required)

Runs the SDM (Spoken Dialogue Model) on the generated voice prompts.

- **Input**: `voice_prompt/${test_data_name}/manifest.jsonl`
- **Output**:
  - `pred_audio/`: Generated response audio files.
  - `pred_text`: The text the model intended to speak.
  - `question_text`: The original input text.

*Note: Ensure your inference script exports both audio and the corresponding text for accurate WER calculation.*

### Step 4: Evaluation
**Script**: `src/batch_score.py`

Calculates metrics for the model outputs. This script orchestrates the following sub-modules:

#### 4.1 ChatGPT Score (Content Quality)
- **Script**: `src/gpt_score.py`
- **Method**: Uses GPT-4o mini (or equivalent) to rate the response (0-100) based on relevance and accuracy compared to the reference.
- **Output**: `chatgpt_score.jsonl`

#### 4.2 ASR-WER (Speech-Text Alignment)
- **Script**: `src/wer.py`
- **Method**: Transcribes `pred_audio` using **Whisper-large-v3** and calculates Word Error Rate (WER) against `pred_text`.
- **Output**: `asr_wer.jsonl`

#### 4.3 UTMOS (Speech Quality)
- **Script**: `src/utmos.py`
- **Method**: Predicts Mean Opinion Score (MOS) for naturalness using UTMOS/VoiceMOS.
- **Output**: `utmos.jsonl`

## Usage

### Prerequisites

Install dependencies:
```bash
pip install openai jiwer torch whisper utmosv2 datasets soundfile
```

Set environment variables for scoring:
```bash
export NEWAPI_API_KEY="your_api_key"
export NEWAPI_BASE_URL="https://api.newapi.com/v1"
export GPT_JUDGE_MODEL="gpt-4o-mini"
```

### Running the Pipeline

1.  **Download Data**:
    ```bash
    python src/download_test_json_from_huggingface.py --dataset Jiann/STORAL --config default --split test --output_dir evaluation/text_prompt
    ```

2.  **Generate Voice Prompts**:
    ```bash
    python src/tts_from_test_jsonl.py --test_data_name Jiann_STORAL_default_storal_en_test
    ```

3.  **Run Inference**:
    (Execute your model's batch inference script pointing to the generated manifest)

4.  **Run Evaluation**:
    Configure `DATASET_DIRS` in `src/batch_score.py` and run:
    ```bash
    python src/batch_score.py
    ```

5.  **View Results**:
    ```bash
    python src/show_results.py
    ```

## Output Format Examples

**`chatgpt_score.jsonl`**:
```json
{
  "id": 1,
  "question_text": "...",
  "pred_text": "...",
  "ref_text": "...",
  "chatgpt_score": 85.0
}
```

**`asr_wer.jsonl`**:
```json
{
  "id": 1,
  "pred_text": "expected output text",
  "transcribed_text": "what the model actually said",
  "wer": 0.02
}
```

**`utmos.jsonl`**:
```json
{
  "id": 1,
  "pred_audio": "/path/to/pred_audio/1.wav",
  "utmos_mos": 4.12
}
```
