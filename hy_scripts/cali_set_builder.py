import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from datasets import load_dataset, concatenate_datasets


# ──────────────────────────────────────────────
# 설정 (import 시점에 사이드이펙트 없음)
# ──────────────────────────────────────────────
TARGET_SR = 16000
SEED      = 42

# 총 512개 구성
# LibriSpeech  : 300 (representative 220 + stress 80)
# Noisy        : 125 (SNR 버킷 3단계 — very_low/low/medium)
# Multilingual : 75  (FLEURS 15개 언어 × 5개)
# No-speech    : 12  (silence 4 + gaussian_noise 4 + bandlimited_noise 4)
N_LIBRISPEECH_REPR   = 220
N_LIBRISPEECH_STRESS = 80
N_NOISY              = 125
N_MULTILINGUAL       = 75
N_NO_SPEECH          = 12
N_TOTAL              = N_LIBRISPEECH_REPR + N_LIBRISPEECH_STRESS + N_NOISY + N_MULTILINGUAL + N_NO_SPEECH

# Noisy SNR 버킷: very_low(0~5dB) 40% / low(5~15dB) 35% / medium(15~20dB) 25%
# 저SNR에 더 많이 배치 — W4A4에서 outlier가 극단 입력에서 집중 발생하기 때문
NOISY_BUCKET_COUNTS = {
    "very_low": int(N_NOISY * 0.40),                                        # 50
    "low":      int(N_NOISY * 0.35),                                        # 43
    "medium":   N_NOISY - int(N_NOISY * 0.40) - int(N_NOISY * 0.35),       # 32
}
SNR_RANGES = {
    "very_low": (0,  5),
    "low":      (5,  15),
    "medium":   (15, 20),
}

# FLEURS 15개 언어 — 어족을 최대한 다양하게 선택
FLEURS_LANGUAGES = [
    "ko_kr",   # 한국어     (교착어)
    "ja_jp",   # 일본어     (교착어)
    "fr_fr",   # 프랑스어   (로망스)
    "de_de",   # 독일어     (게르만)
    "es_es",   # 스페인어   (로망스)
    "ar_eg",   # 아랍어     (셈어족)
    "hi_in",   # 힌디어     (인도유럽)
    "ru_ru",   # 러시아어   (슬라브)
    "tr_tr",   # 터키어     (튀르크)
    "vi_vn",   # 베트남어   (오스트로아시아)
    "th_th",   # 태국어     (타이카다이)
    "id_id",   # 인도네시아어 (오스트로네시아)
    "pl_pl",   # 폴란드어   (슬라브)
    "pt_br",   # 포르투갈어 (로망스)
    "nl_nl",   # 네덜란드어 (게르만)
]
N_PER_LANGUAGE = N_MULTILINGUAL // len(FLEURS_LANGUAGES)  # 5

LIBRISPEECH_SPLITS = ["test.clean", "test.other"]

DURATION_BUCKETS = {
    "short":  (0.0,  3.0),
    "medium": (3.0,  8.0),
    "long":   (8.0, 20.0),
}
MAX_PER_SPEAKER = 3


# ──────────────────────────────────────────────
# 공통 오디오 유틸
# ──────────────────────────────────────────────

def resample_if_needed(audio_array: np.ndarray, sr: int,
                       target_sr: int = TARGET_SR):
    if sr == target_sr:
        return audio_array.astype(np.float32), sr
    waveform_t = torch.from_numpy(audio_array.astype(np.float32)).unsqueeze(0)
    waveform_t = torchaudio.functional.resample(waveform_t, sr, target_sr)
    return waveform_t.squeeze(0).numpy(), target_sr


def get_duration(waveform: np.ndarray, sr: int = TARGET_SR) -> float:
    return len(waveform) / sr


def compute_rms_energy(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform)) + 1e-12))


def compute_zero_crossing_rate(waveform: np.ndarray) -> float:
    if len(waveform) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.signbit(waveform).astype(np.int8)))))


def compute_spectral_centroid(waveform: np.ndarray,
                               sr: int = TARGET_SR) -> float:
    if len(waveform) == 0:
        return 0.0
    segment = waveform[: min(len(waveform), sr)]
    fft_mag = np.abs(np.fft.rfft(segment))
    freqs   = np.fft.rfftfreq(len(segment), d=1.0 / sr)
    denom   = np.sum(fft_mag) + 1e-12
    return float(np.sum(freqs * fft_mag) / denom)


def compute_frame_energy_stats(waveform: np.ndarray, sr: int = TARGET_SR):
    frame_len = int(0.025 * sr)
    hop_len   = int(0.010 * sr)
    if len(waveform) < frame_len:
        e = float(np.sum(np.square(waveform)))
        return {"energy_p10": e, "energy_p50": e,
                "energy_p90": e, "energy_dyn_range": 0.0}
    energies = [
        np.sum(np.square(waveform[i: i + frame_len]))
        for i in range(0, len(waveform) - frame_len + 1, hop_len)
    ]
    energies = np.asarray(energies, dtype=np.float32)
    p10 = float(np.percentile(energies, 10))
    p50 = float(np.percentile(energies, 50))
    p90 = float(np.percentile(energies, 90))
    dyn = float(np.log10((p90 + 1e-12) / (p10 + 1e-12) + 1e-12))
    return {"energy_p10": p10, "energy_p50": p50,
            "energy_p90": p90, "energy_dyn_range": dyn}


def compute_text_features(transcript: str, duration: float):
    words   = transcript.strip().split()
    n_words = len(words)
    n_chars = len(transcript.replace(" ", ""))
    return {
        "n_words":         n_words,
        "n_chars":         n_chars,
        "speech_rate_wps": float(n_words / max(duration, 1e-6)),
        "chars_per_sec":   float(n_chars / max(duration, 1e-6)),
        "sec_per_word":    float(duration / max(n_words, 1)),
    }


def assign_duration_bucket(duration: float):
    for name, (lo, hi) in DURATION_BUCKETS.items():
        if lo <= duration < hi:
            return name
    return None


def make_record(global_idx, sample_id, speaker_id, chapter_id,
                source_split, transcript, audio_array, sr,
                subset, language="en", extra=None):
    """
    실제 audio_array로 모든 feature를 계산해서 레코드를 만드는 공통 함수.
    extra: dict — subset별 추가 필드 (snr_db, snr_bucket, no_speech_type 등)
    """
    duration = get_duration(audio_array, sr)
    bucket   = assign_duration_bucket(duration)
    if bucket is None:
        return None

    spk_id = speaker_id
    if not isinstance(spk_id, int):
        s = str(spk_id).lstrip("-")
        spk_id = int(s) if s.isdigit() else hash(str(spk_id)) % 100000

    rec = {
        "global_idx":        global_idx,
        "id":                str(sample_id),
        "speaker_id":        spk_id,
        "chapter_id":        int(chapter_id) if str(chapter_id).lstrip("-").isdigit() else -1,
        "source_split":      source_split,
        "language":          language,
        "subset":            subset,
        "transcript":        transcript,
        "duration":          float(duration),
        "bucket":            bucket,
        "rms_energy":        compute_rms_energy(audio_array),
        "zcr":               compute_zero_crossing_rate(audio_array),
        "spectral_centroid": compute_spectral_centroid(audio_array, sr),
    }
    rec.update(compute_text_features(transcript, duration))
    rec.update(compute_frame_energy_stats(audio_array, sr))
    rec["audio_samples_per_word"] = float(len(audio_array) / max(rec["n_words"], 1))
    rec["audio_sec_per_char"]     = float(duration / max(rec["n_chars"], 1))
    if extra:
        rec.update(extra)
    return rec


# ──────────────────────────────────────────────
# 샘플링 유틸
# ──────────────────────────────────────────────

def quantile_bin_indices(values, n_bins=4):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return np.array([], dtype=np.int64)
    edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    edges = np.maximum.accumulate(edges)
    return np.digitize(values, edges[1:-1], right=False)


def sample_diverse_group(items, n_select, max_per_speaker=MAX_PER_SPEAKER):
    if len(items) <= n_select:
        return items[:]
    feat_names = ["rms_energy", "spectral_centroid",
                  "speech_rate_wps", "audio_samples_per_word"]
    bin_keys = [quantile_bin_indices([x[f] for x in items]) for f in feat_names]
    for i, item in enumerate(items):
        item["_bin_key"] = tuple(k[i] for k in bin_keys)

    groups = defaultdict(list)
    for item in items:
        groups[item["_bin_key"]].append(item)
    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    for k in group_keys:
        random.shuffle(groups[k])

    speaker_count = defaultdict(int)
    selected      = []
    made_progress = True
    while len(selected) < n_select and made_progress:
        made_progress = False
        for k in group_keys:
            while groups[k]:
                c = groups[k].pop()
                if speaker_count[c["speaker_id"]] < max_per_speaker:
                    selected.append(c)
                    speaker_count[c["speaker_id"]] += 1
                    made_progress = True
                    break
            if len(selected) >= n_select:
                break

    if len(selected) < n_select:
        remaining = [x for x in items if x not in selected]
        random.shuffle(remaining)
        for c in remaining:
            if speaker_count[c["speaker_id"]] < max_per_speaker:
                selected.append(c)
                speaker_count[c["speaker_id"]] += 1
            if len(selected) >= n_select:
                break

    if len(selected) < n_select:
        remaining = [x for x in items if x not in selected]
        selected.extend(remaining[: n_select - len(selected)])

    for item in items:
        item.pop("_bin_key", None)
    return selected[:n_select]


def top_k_unique(items, key_name, k, reverse=True, exclude_ids=None):
    exclude_ids = exclude_ids or set()
    pool = sorted(
        [x for x in items if x["id"] not in exclude_ids],
        key=lambda x: x[key_name], reverse=reverse,
    )
    speaker_count = defaultdict(int)
    selected = []
    for x in pool:
        if speaker_count[x["speaker_id"]] >= MAX_PER_SPEAKER:
            continue
        selected.append(x)
        speaker_count[x["speaker_id"]] += 1
        if len(selected) >= k:
            break
    return selected


# ──────────────────────────────────────────────
# 1. LibriSpeech
# ──────────────────────────────────────────────

def load_librispeech():
    ds_list = []
    for split in LIBRISPEECH_SPLITS:
        print(f"  Loading librispeech_asr [{split}] ...")
        ds = load_dataset("openslr/librispeech_asr", split=split,
                          trust_remote_code=True)
        ds = ds.add_column("source_split", [split] * len(ds))
        ds_list.append(ds)
    return concatenate_datasets(ds_list)


def build_librispeech_records(dataset):
    """dataset을 순회하며 실제 오디오로 feature를 계산."""
    print(f"  Building LibriSpeech records ({len(dataset)} samples) ...")
    records = []
    for idx, sample in enumerate(dataset):
        if idx % 500 == 0:
            print(f"    {idx}/{len(dataset)}")
        audio, sr = resample_if_needed(
            sample["audio"]["array"], sample["audio"]["sampling_rate"]
        )
        rec = make_record(
            global_idx=idx,
            sample_id=sample.get("id", idx),
            speaker_id=sample.get("speaker_id", -1),
            chapter_id=sample.get("chapter_id", -1),
            source_split=sample.get("source_split", "unknown"),
            transcript=sample["text"],
            audio_array=audio, sr=sr,
            subset="librispeech",
            language="en",
        )
        if rec:
            records.append(rec)
    print(f"  Valid records: {len(records)}")
    return records


def build_librispeech_representative(records, n):
    by_bucket  = defaultdict(list)
    for r in records:
        by_bucket[r["bucket"]].append(r)
    per_bucket = {
        "short":  n // 3,
        "medium": n // 3,
        "long":   n - 2 * (n // 3),
    }
    selected = []
    for bname, nb in per_bucket.items():
        chosen = sample_diverse_group(list(by_bucket[bname]), nb)
        print(f"    {bname}: {len(chosen)}")
        selected.extend(chosen)
    return selected


def build_librispeech_stress(records, n, exclude_ids=None):
    exclude_ids = exclude_ids or set()
    metrics = [
        ("duration",              True),
        ("rms_energy",            True),
        ("speech_rate_wps",       True),
        ("audio_samples_per_word",True),
        ("energy_dyn_range",      True),
        ("zcr",                   True),
        ("spectral_centroid",     True),
    ]
    per_metric  = max(1, n // len(metrics))
    selected    = []
    selected_ids = set(exclude_ids)

    for key_name, reverse in metrics:
        chunk = top_k_unique(records, key_name, per_metric, reverse, selected_ids)
        selected.extend(chunk)
        selected_ids.update(x["id"] for x in chunk)
        print(f"    metric={key_name:22s} -> {len(chunk)}")

    if len(selected) < n:
        remaining = [x for x in records if x["id"] not in selected_ids]
        keys  = [m[0] for m in metrics]
        vals  = {k: np.array([r[k] for r in remaining], dtype=np.float32) for k in keys}
        stats = {k: (vals[k].mean(), vals[k].std() + 1e-6) for k in keys}
        scored = sorted(
            [(sum(abs((r[k] - stats[k][0]) / stats[k][1]) for k in keys), r)
             for r in remaining],
            key=lambda x: x[0], reverse=True,
        )
        speaker_count = defaultdict(int)
        for _, r in scored:
            if speaker_count[r["speaker_id"]] >= MAX_PER_SPEAKER:
                continue
            selected.append(r)
            speaker_count[r["speaker_id"]] += 1
            if len(selected) >= n:
                break

    return selected[:n]


# ──────────────────────────────────────────────
# 2. Noisy augmentation  ← 핵심 수정 부분
# ──────────────────────────────────────────────

def add_noise(audio: np.ndarray, snr_db: float,
              rng: np.random.Generator) -> np.ndarray:
    """가우시안 노이즈를 지정 SNR(dB)로 혼합 후 클리핑."""
    signal_power = np.mean(audio ** 2) + 1e-12
    noise        = rng.standard_normal(len(audio)).astype(np.float32)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = noise * np.sqrt(noise_power / (np.mean(noise ** 2) + 1e-12))
    return np.clip(audio + noise, -1.0, 1.0)


def build_noisy_records(dataset, n_total):
    """
    dataset(HuggingFace Dataset)에서 실제 오디오를 꺼내
    SNR augmentation 후 feature를 새로 계산.

    SNR 버킷별 샘플 수:
      very_low (0~5 dB)  : 50개  — 극단 노이즈, outlier 포착 핵심
      low      (5~15 dB) : 43개  — 중간 노이즈
      medium   (15~20 dB): 32개  — 약한 노이즈
    """
    print(f"  Building noisy records (n={n_total}) ...")
    rng = np.random.default_rng(SEED + 1)

    # duration 버킷별로 pool을 나눠서 순서 섞기 (다양성 유지)
    by_duration: dict[str, list[int]] = defaultdict(list)
    for i, sample in enumerate(dataset):
        dur = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
        b   = assign_duration_bucket(dur)
        if b:
            by_duration[b].append(i)
    for b in by_duration:
        random.shuffle(by_duration[b])

    # 각 버킷에서 round-robin으로 index를 꺼내는 이터레이터
    dur_iters = {b: iter(idxs) for b, idxs in by_duration.items()}
    dur_cycle = ["short", "medium", "long"]

    def next_dataset_index():
        """duration 버킷을 순환하며 다음 dataset index를 반환."""
        for _ in range(len(dur_cycle) * 3):
            b = dur_cycle[next_dataset_index._ptr % len(dur_cycle)]
            next_dataset_index._ptr += 1
            try:
                return next(dur_iters[b])
            except StopIteration:
                continue
        return None
    next_dataset_index._ptr = 0

    records      = []
    global_base  = 900000
    used_speakers: dict[str, int] = defaultdict(int)  # snr_bucket -> speaker 사용 횟수

    for bucket_name, n_bucket in NOISY_BUCKET_COUNTS.items():
        snr_lo, snr_hi = SNR_RANGES[bucket_name]
        added = 0
        speaker_count: dict[int, int] = defaultdict(int)

        while added < n_bucket:
            ds_idx = next_dataset_index()
            if ds_idx is None:
                print(f"    WARNING: pool 소진, {bucket_name} {added}/{n_bucket}만 생성")
                break

            sample = dataset[ds_idx]
            spk_id_raw = sample.get("speaker_id", -1)
            spk_id = int(spk_id_raw) if str(spk_id_raw).lstrip("-").isdigit() \
                     else hash(str(spk_id_raw)) % 100000

            if speaker_count[spk_id] >= MAX_PER_SPEAKER:
                continue

            # 실제 오디오 로드 및 리샘플
            audio, sr = resample_if_needed(
                sample["audio"]["array"], sample["audio"]["sampling_rate"]
            )

            # SNR을 버킷 범위 내에서 균등 샘플링
            snr_db = float(rng.uniform(snr_lo, snr_hi))

            # 실제 노이즈를 오디오에 적용
            noisy_audio = add_noise(audio, snr_db, rng)

            # noisy_audio로 feature 재계산
            rec = make_record(
                global_idx=global_base + len(records),
                sample_id=f"noisy_{sample.get('id', ds_idx)}_{bucket_name}_{added}",
                speaker_id=spk_id,
                chapter_id=sample.get("chapter_id", -1),
                source_split=sample.get("source_split", "unknown"),
                transcript=sample["text"],
                audio_array=noisy_audio,
                sr=sr,
                subset="noisy",
                language="en",
                extra={
                    "snr_db":     snr_db,
                    "snr_bucket": bucket_name,
                    "source_id":  str(sample.get("id", ds_idx)),
                },
            )
            if rec is None:
                continue

            records.append(rec)
            speaker_count[spk_id] += 1
            added += 1

        print(f"    SNR bucket '{bucket_name}' ({snr_lo}~{snr_hi} dB): {added}")

    print(f"  Total noisy records: {len(records)}")
    return records


# ──────────────────────────────────────────────
# 3. Multilingual (FLEURS)
# ──────────────────────────────────────────────

def load_fleurs_language(lang_code: str, n_samples: int, lang_idx: int):
    try:
        ds = load_dataset("google/fleurs", lang_code,
                          split="test", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("google/fleurs", lang_code,
                              split="validation", trust_remote_code=True)
        except Exception as e:
            print(f"    [{lang_code}] 로드 실패: {e}")
            return []

    indices = list(range(len(ds)))
    random.shuffle(indices)
    records     = []
    global_base = 800000 + lang_idx * 100

    for i, idx in enumerate(indices):
        if len(records) >= n_samples:
            break
        sample     = ds[idx]
        audio_arr  = np.array(sample["audio"]["array"], dtype=np.float32)
        audio, sr  = resample_if_needed(audio_arr, sample["audio"]["sampling_rate"])
        transcript = sample.get("transcription",
                     sample.get("raw_transcription", ""))
        if not transcript:
            continue

        rec = make_record(
            global_idx=global_base + i,
            sample_id=f"fleurs_{lang_code}_{idx}",
            speaker_id=sample.get("speaker_id", sample.get("id", i)),
            chapter_id=-1,
            source_split=f"fleurs_{lang_code}",
            transcript=transcript,
            audio_array=audio, sr=sr,
            subset="multilingual",
            language=lang_code,
        )
        if rec:
            records.append(rec)

    return records


def build_multilingual_records():
    print(f"  Building multilingual records "
          f"({len(FLEURS_LANGUAGES)} languages × {N_PER_LANGUAGE}) ...")
    all_records = []
    for lang_idx, lang in enumerate(FLEURS_LANGUAGES):
        recs = load_fleurs_language(lang, N_PER_LANGUAGE, lang_idx)
        print(f"    [{lang}]: {len(recs)} records")
        all_records.extend(recs)
    print(f"  Total multilingual: {len(all_records)}")
    return all_records


# ──────────────────────────────────────────────
# 4. No-speech (synthetic)
# ──────────────────────────────────────────────

def build_no_speech_records():
    """
    3가지 타입 × 4가지 길이 = 12개.
    모두 실제 합성 오디오를 만들어서 feature를 계산.

    - silence          : quantization floor 수준의 미세 노이즈만 (완전 무음)
    - gaussian_noise   : 백색 잡음 (RMS 0.01~0.05)
    - bandlimited_noise: 4000~8000 Hz 대역 노이즈 (speech 대역 회피)
                         → AuT encoder가 non-speech embedding을 출력하는 경로 커버
    """
    print(f"  Building no-speech records (n={N_NO_SPEECH}) ...")
    rng           = np.random.default_rng(SEED + 99)
    records       = []
    durations_sec = [1.0, 2.5, 5.0, 10.0]

    def _make_no_speech_rec(audio, ns_type, idx_local):
        """no-speech는 transcript가 없으므로 text feature를 0으로 처리."""
        duration = get_duration(audio)
        ef       = compute_frame_energy_stats(audio)
        return {
            "global_idx":             700000 + idx_local,
            "id":                     f"no_speech_{ns_type}_{idx_local}",
            "speaker_id":             -1,
            "chapter_id":             -1,
            "source_split":           f"synthetic_{ns_type}",
            "language":               "none",
            "subset":                 "no_speech",
            "no_speech_type":         ns_type,
            "transcript":             "",
            "duration":               float(duration),
            "bucket":                 assign_duration_bucket(duration) or "medium",
            "rms_energy":             compute_rms_energy(audio),
            "zcr":                    compute_zero_crossing_rate(audio),
            "spectral_centroid":      compute_spectral_centroid(audio),
            "n_words":                0,
            "n_chars":                0,
            "speech_rate_wps":        0.0,
            "chars_per_sec":          0.0,
            "sec_per_word":           0.0,
            "audio_samples_per_word": float(len(audio)),
            "audio_sec_per_char":     0.0,
            **ef,
        }

    # Type 1: silence
    for i, dur in enumerate(durations_sec):
        n_samp = int(dur * TARGET_SR)
        audio  = rng.uniform(-1e-4, 1e-4, n_samp).astype(np.float32)
        records.append(_make_no_speech_rec(audio, "silence", i))

    # Type 2: white gaussian noise
    for i, dur in enumerate(durations_sec):
        n_samp     = int(dur * TARGET_SR)
        rms_target = float(rng.uniform(0.01, 0.05))
        audio      = rng.standard_normal(n_samp).astype(np.float32)
        audio      = audio / (np.sqrt(np.mean(audio ** 2)) + 1e-12) * rms_target
        records.append(_make_no_speech_rec(audio, "gaussian_noise", 4 + i))

    # Type 3: bandlimited noise (4000~8000 Hz)
    for i, dur in enumerate(durations_sec):
        n_samp  = int(dur * TARGET_SR)
        white   = rng.standard_normal(n_samp).astype(np.float32)
        fft_val = np.fft.rfft(white)
        freqs   = np.fft.rfftfreq(n_samp, d=1.0 / TARGET_SR)
        fft_val[~((freqs >= 4000) & (freqs <= 8000))] = 0.0
        audio   = np.fft.irfft(fft_val, n=n_samp).astype(np.float32)
        rms_target = float(rng.uniform(0.01, 0.03))
        audio   = audio / (np.sqrt(np.mean(audio ** 2)) + 1e-12) * rms_target
        records.append(_make_no_speech_rec(audio, "bandlimited_noise", 8 + i))

    print(f"  No-speech records: {len(records)}")
    return records


# ──────────────────────────────────────────────
# 저장 및 요약
# ──────────────────────────────────────────────

def save_outputs(selected: list, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)

    meta_path = out_dir / "calibration_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    idx_path = out_dir / "calibration_indices.txt"
    with open(idx_path, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(f"{r['global_idx']}\n")

    # subset별 인덱스 분리 저장
    subsets = sorted(set(r["subset"] for r in selected))
    for s in subsets:
        with open(out_dir / f"indices_{s}.txt", "w", encoding="utf-8") as f:
            for r in selected:
                if r["subset"] == s:
                    f.write(f"{r['global_idx']}\n")

    # ── 요약 출력 ──
    print("\n" + "=" * 70)
    print("CALIBRATION SET SUMMARY")
    print("=" * 70)
    print(f"Total samples   : {len(selected)}")
    print(f"Unique speakers : {len(set(r['speaker_id'] for r in selected))}")

    print("\nSubset distribution:")
    for s in subsets:
        cnt = sum(1 for r in selected if r["subset"] == s)
        print(f"  {s:20s}: {cnt}")

    print("\nLanguage distribution:")
    lang_cnt: dict[str, int] = defaultdict(int)
    for r in selected:
        lang_cnt[r["language"]] += 1
    for lang, cnt in sorted(lang_cnt.items(), key=lambda x: -x[1]):
        print(f"  {lang:12s}: {cnt}")

    print("\nDuration bucket distribution:")
    for b in DURATION_BUCKETS:
        cnt = sum(1 for r in selected if r.get("bucket") == b)
        print(f"  {b:10s}: {cnt}")

    print("\nNoisy SNR bucket distribution:")
    noisy = [r for r in selected if r["subset"] == "noisy"]
    for nb in ["very_low", "low", "medium"]:
        cnt    = sum(1 for r in noisy if r.get("snr_bucket") == nb)
        lo, hi = SNR_RANGES[nb]
        print(f"  {nb:12s} ({lo:2d}~{hi:2d} dB): {cnt}")

    print("\nNo-speech type distribution:")
    for t in ["silence", "gaussian_noise", "bandlimited_noise"]:
        cnt = sum(1 for r in selected
                  if r["subset"] == "no_speech" and r.get("no_speech_type") == t)
        print(f"  {t:20s}: {cnt}")

    print("\nAcoustic feature stats (all subsets):")
    feat_keys = ["duration", "rms_energy", "zcr", "spectral_centroid",
                 "speech_rate_wps", "audio_samples_per_word", "energy_dyn_range"]
    for k in feat_keys:
        arr = np.array([r[k] for r in selected if r.get(k) is not None],
                       dtype=np.float32)
        print(f"  {k:22s}  mean={arr.mean():8.4f}  std={arr.std():8.4f}"
              f"  min={arr.min():8.4f}  max={arr.max():8.4f}")

    print(f"\nSaved → {meta_path}")
    print(f"Saved → {idx_path}")
    for s in subsets:
        print(f"Saved → {out_dir}/indices_{s}.txt")


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────

def build_cali_set(
    output_dir: str = "./calibration_set",
    seed: int = SEED,
):
    """
    Calibration set을 빌드하고 output_dir에 저장.

    Parameters
    ----------
    output_dir : str
        결과물을 저장할 디렉토리 경로.
    seed : int
        재현성을 위한 랜덤 시드.

    Example
    -------
    >>> from build_calibration_set import main
    >>> main(output_dir="./my_calib", seed=42)
    """
    # 사이드이펙트를 main() 호출 시점으로 한정
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    random.seed(seed)
    np.random.seed(seed)

    assert N_TOTAL == 512, f"총합 오류: {N_TOTAL}"

    print("=" * 70)
    print(f"Target composition  total={N_TOTAL}")
    print(f"  LibriSpeech repr  : {N_LIBRISPEECH_REPR}")
    print(f"  LibriSpeech stress: {N_LIBRISPEECH_STRESS}")
    print(f"  Noisy augmented   : {N_NOISY}  "
          f"(very_low={NOISY_BUCKET_COUNTS['very_low']}, "
          f"low={NOISY_BUCKET_COUNTS['low']}, "
          f"medium={NOISY_BUCKET_COUNTS['medium']})")
    print(f"  Multilingual      : {N_MULTILINGUAL}  "
          f"({len(FLEURS_LANGUAGES)} langs × {N_PER_LANGUAGE})")
    print(f"  No-speech         : {N_NO_SPEECH}  (3 types × 4 durations)")
    print(f"  output_dir        : {out_dir}")
    print(f"  seed              : {seed}")
    print("=" * 70)

    # ── 1. LibriSpeech ──
    print("\n[1/4] LibriSpeech ...")
    ls_dataset = load_librispeech()
    ls_records = build_librispeech_records(ls_dataset)

    print("  Sampling representative ...")
    ls_repr  = build_librispeech_representative(ls_records, N_LIBRISPEECH_REPR)
    repr_ids = set(r["id"] for r in ls_repr)

    print("  Sampling stress ...")
    ls_stress = build_librispeech_stress(ls_records, N_LIBRISPEECH_STRESS,
                                         exclude_ids=repr_ids)

    # ── 2. Noisy — ls_dataset을 직접 넘겨 실제 오디오에 노이즈 적용 ──
    print("\n[2/4] Noisy augmentation ...")
    noisy_records = build_noisy_records(ls_dataset, N_NOISY)

    # ── 3. Multilingual ──
    print("\n[3/4] Multilingual (FLEURS) ...")
    ml_records = build_multilingual_records()

    # ── 4. No-speech ──
    print("\n[4/4] No-speech (synthetic) ...")
    ns_records = build_no_speech_records()

    # ── 합치기 ──
    selected = ls_repr + ls_stress + noisy_records + ml_records + ns_records
    print(f"\nTotal assembled: {len(selected)}  (target: {N_TOTAL})")
    if len(selected) > N_TOTAL:
        print(f"  Trimming {len(selected) - N_TOTAL} excess samples ...")
        selected = selected[:N_TOTAL]

    save_outputs(selected, out_dir)
    print("\nDone.")