# pip install -U torch transformers datasets evaluate jiwer soundfile librosa
# pip install -U qwen-asr

import re
import csv
import json
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import evaluate
from datasets import load_dataset, Audio

from qwen_asr import Qwen3ASRModel


# =========================================================
# Config
# =========================================================
MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"

# Hugging Face dataset 예시
USE_HF_DATASET = True
HF_DATASET_NAME = "librispeech_asr"
HF_DATASET_CONFIG = "clean"
HF_DATASET_SPLIT = "test[:100]"

# 로컬 데이터셋 예시 (USE_HF_DATASET=False일 때 사용)
# metadata.csv 형식:
# path,text
# /path/to/a.wav,hello world
LOCAL_METADATA_CSV = "metadata.csv"

AUDIO_COLUMN = "audio"
TEXT_COLUMN = "text"

FORCE_LANGUAGE = "English"   # 자동 언어 인식 쓰려면 None
BATCH_SIZE = 8
MAX_NEW_TOKENS = 256

COMPUTE_CER = True
SAVE_PREDICTIONS = True
PRED_SAVE_PATH = "qwen3_asr_eval_predictions.jsonl"


# =========================================================
# Text normalization
# =========================================================
def normalize_text(text: str) -> str:
    """
    WER/CER 계산용 간단 정규화.
    영어 기준으로 무난한 버전.
    벤치마크 재현이 목적이면 해당 데이터셋의 공식 normalize 규칙에 맞춰 바꿔야 함.
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()

    # 영어용 단순 punctuation 제거
    text = re.sub(r"[^\w\s']", " ", text)

    # 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================================================
# Data loading
# =========================================================
def load_hf_samples() -> List[Dict[str, Any]]:
    ds = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split=HF_DATASET_SPLIT)
    ds = ds.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))

    samples = []
    for ex in ds:
        samples.append(
            {
                "id": ex.get("id", None),
                "audio": (ex[AUDIO_COLUMN]["array"], ex[AUDIO_COLUMN]["sampling_rate"]),
                "reference": ex[TEXT_COLUMN],
                "source": "hf",
            }
        )
    return samples


def load_local_samples() -> List[Dict[str, Any]]:
    """
    metadata.csv:
    path,text
    a.wav,hello world
    b.wav,this is a test
    """
    samples = []
    with open(LOCAL_METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            samples.append(
                {
                    "id": row.get("id", str(i)),
                    "audio": row["path"],     # qwen-asr는 path 입력 가능
                    "reference": row["text"],
                    "source": "local",
                }
            )
    return samples


def load_samples() -> List[Dict[str, Any]]:
    if USE_HF_DATASET:
        return load_hf_samples()
    return load_local_samples()


# =========================================================
# Utils
# =========================================================
def batchify(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# =========================================================
# Main evaluation
# =========================================================
def main():
    print("Loading metrics...")
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer") if COMPUTE_CER else None

    print("Loading samples...")
    samples = load_samples()
    print(f"Loaded {len(samples)} samples")

    print("Loading model...")
    model = Qwen3ASRModel.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        max_inference_batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    predictions = []
    references = []
    raw_predictions = []
    raw_references = []
    records = []

    print("Running inference...")
    for batch in batchify(samples, BATCH_SIZE):
        batch_audio = [x["audio"] for x in batch]
        batch_refs = [x["reference"] for x in batch]

        # context는 scalar로 broadcast 가능
        # language도 scalar로 broadcast 가능
        results = model.transcribe(
            audio=batch_audio,
            context="",
            language=FORCE_LANGUAGE,
            return_time_stamps=False,
        )

        for sample, result, ref in zip(batch, results, batch_refs):
            pred_raw = result.text
            ref_raw = ref

            pred_norm = normalize_text(pred_raw)
            ref_norm = normalize_text(ref_raw)

            raw_predictions.append(pred_raw)
            raw_references.append(ref_raw)

            predictions.append(pred_norm)
            references.append(ref_norm)

            records.append(
                {
                    "id": sample["id"],
                    "source": sample["source"],
                    "language_pred": result.language,
                    "prediction_raw": pred_raw,
                    "reference_raw": ref_raw,
                    "prediction_norm": pred_norm,
                    "reference_norm": ref_norm,
                }
            )

    print("Computing metrics...")
    wer = wer_metric.compute(predictions=predictions, references=references)

    print(f"\nTotal samples: {len(predictions)}")
    print(f"WER: {wer:.4f} ({wer * 100:.2f}%)")

    if COMPUTE_CER:
        cer = cer_metric.compute(predictions=predictions, references=references)
        print(f"CER: {cer:.4f} ({cer * 100:.2f}%)")

    if SAVE_PREDICTIONS:
        out_path = Path(PRED_SAVE_PATH)
        with out_path.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved predictions to: {out_path.resolve()}")

    print("\nExamples:")
    for i in range(min(5, len(records))):
        print("-" * 80)
        print(f"ID   : {records[i]['id']}")
        print(f"LANG : {records[i]['language_pred']}")
        print(f"REF  : {records[i]['reference_norm']}")
        print(f"HYP  : {records[i]['prediction_norm']}")


if __name__ == "__main__":
    main()