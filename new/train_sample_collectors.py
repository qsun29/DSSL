"""
各数据集训练样本收集：根据 data_root 目录结构识别数据集并返回 (路径, 标签) 列表。
供 long_wavlm_disease_mask_experiment 与 search_pathology_weights 使用。
"""

import csv as csv_module
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def collect_adress_train_samples(data_root: str) -> List[Tuple[str, int]]:
    """ADReSS: ADReSS-IS2020-data/train/Full_wave_enhanced_audio/ cc=HC(1), cd=AD(0)."""
    root = Path(data_root)
    audio_root = root / "ADReSS-IS2020-data" / "train" / "Full_wave_enhanced_audio"
    if not audio_root.exists():
        raise FileNotFoundError(f"ADReSS train audio not found: {audio_root}")
    samples: List[Tuple[str, int]] = []
    for d, label in [(audio_root / "cd", 0), (audio_root / "cc", 1)]:
        if d.exists():
            for wav in sorted(d.glob("*.wav")):
                samples.append((str(wav.resolve()), label))
    if not samples:
        raise RuntimeError(f"No wav files under {audio_root}")
    logger.info(f"Collected {len(samples)} samples from ADReSS train")
    return samples


def collect_adresso21_train_samples(data_root: str) -> List[Tuple[str, int]]:
    """ADReSSo21: diagnosis/train/audio/ ad=AD(0), cn=HC(1)."""
    root = Path(data_root)
    audio_root = root / "diagnosis" / "train" / "audio"
    if not audio_root.exists():
        raise FileNotFoundError(f"ADReSSo21 train audio not found: {audio_root}")
    samples: List[Tuple[str, int]] = []
    for d, label in [(audio_root / "ad", 0), (audio_root / "cn", 1)]:
        if d.exists():
            for wav in sorted(d.glob("*.wav")):
                samples.append((str(wav.resolve()), label))
    if not samples:
        raise RuntimeError(f"No wav files under {audio_root}")
    logger.info(f"Collected {len(samples)} samples from ADReSSo21 diagnosis train")
    return samples


def collect_edaic_train_samples(data_root: str) -> List[Tuple[str, int]]:
    """E-DAIC: labels/train_split.csv + data/<Participant_ID>_P/, PHQ_Binary -> label."""
    root = Path(data_root)
    labels_path = root / "labels" / "train_split.csv"
    data_dir = root / "data"
    if not labels_path.exists() or not data_dir.exists():
        raise FileNotFoundError(f"E-DAIC labels or data not found")
    samples: List[Tuple[str, int]] = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        for row in csv_module.DictReader(f):
            try:
                pid = row["Participant_ID"].strip()
                label = 1 - int(row["PHQ_Binary"])
            except (KeyError, ValueError):
                continue
            part_dir = data_dir / f"{pid}_P"
            wavs = sorted(part_dir.glob("*.wav")) if part_dir.exists() else []
            if wavs:
                samples.append((str(wavs[0].resolve()), label))
    if not samples:
        raise RuntimeError(f"No wav under {data_dir}")
    logger.info(f"Collected {len(samples)} samples from E-DAIC train")
    return samples


def collect_speechwellness1_train_samples(data_root: str) -> List[Tuple[str, int]]:
    """SpeechWellness1: SW1-train/{id}/*.wav + Metadata_train.csv (id, label)."""
    root = Path(data_root)
    train_dir = root / "SW1-train"
    meta_path = train_dir / "Metadata_train.csv"
    if not train_dir.exists() or not meta_path.exists():
        raise FileNotFoundError(f"SpeechWellness1 train or metadata not found")
    id_to_label: Dict[str, int] = {}
    with open(meta_path, newline="", encoding="utf-8") as f:
        for row in csv_module.DictReader(f):
            try:
                id_to_label[row["id"].strip()] = int(row["label"])
            except (KeyError, ValueError):
                continue
    samples: List[Tuple[str, int]] = []
    for sid, label in id_to_label.items():
        sub_dir = train_dir / sid
        if sub_dir.is_dir():
            for wav in sorted(sub_dir.glob("*.wav")):
                samples.append((str(wav.resolve()), label))
    if not samples:
        raise RuntimeError(f"No wav under {train_dir}")
    logger.info(f"Collected {len(samples)} samples from SpeechWellness1 SW1-train")
    return samples


def collect_adressm_train_samples(data_root: str) -> List[Tuple[str, int]]:
    """ADReSS-M: train/*.mp3 + training-groundtruth.csv, dx: Control=1, ProbableAD=0."""
    root = Path(data_root)
    train_dir = root / "train"
    meta_path = root / "training-groundtruth.csv"
    if not train_dir.exists() or not meta_path.exists():
        raise FileNotFoundError(f"ADReSS-M train or metadata not found")
    dx_to_label = {"Control": 1, "ProbableAD": 0}
    samples: List[Tuple[str, int]] = []
    with open(meta_path, newline="", encoding="utf-8") as f:
        for row in csv_module.DictReader(f):
            try:
                fname = row["adressfname"].strip().strip('"')
                dx = row["dx"].strip().strip('"')
            except KeyError:
                continue
            if dx not in dx_to_label:
                continue
            for ext in (".mp3", ".wav"):
                wav_path = train_dir / f"{fname}{ext}"
                if wav_path.exists():
                    samples.append((str(wav_path.resolve()), dx_to_label[dx]))
                    break
    if not samples:
        raise RuntimeError(f"No audio under {train_dir}")
    logger.info(f"Collected {len(samples)} samples from ADReSS-M train")
    return samples


def collect_androids_reading_train_samples(data_root: str) -> List[Tuple[str, int]]:
    """Androids-Corpus: Reading-Task/audio/ PT=0, HC=1."""
    root = Path(data_root)
    audio_root = root / "Reading-Task" / "audio"
    hc_dir, pt_dir = audio_root / "HC", audio_root / "PT"
    if not audio_root.exists() or (not hc_dir.exists() and not pt_dir.exists()):
        raise FileNotFoundError(f"Androids Reading-Task audio not found")
    samples: List[Tuple[str, int]] = []
    for d, label in [(pt_dir, 0), (hc_dir, 1)]:
        if d.exists():
            for wav in sorted(d.glob("*.wav")):
                samples.append((str(wav.resolve()), label))
    if not samples:
        raise RuntimeError(f"No wav under {audio_root}")
    logger.info(f"Collected {len(samples)} samples from Androids-Corpus Reading-Task")
    return samples


def collect_neurovoz_train_samples(data_root: str) -> List[Tuple[str, int]]:
    """NeuroVoz: audios/*.wav, PD_*=0, HC_*=1."""
    root = Path(data_root)
    audios_dir = root / "audios"
    if not audios_dir.exists():
        raise FileNotFoundError(f"NeuroVoz audios not found: {audios_dir}")
    samples: List[Tuple[str, int]] = []
    for wav in sorted(audios_dir.glob("*.wav")):
        stem = wav.stem
        if stem.startswith("PD_"):
            samples.append((str(wav.resolve()), 0))
        elif stem.startswith("HC_"):
            samples.append((str(wav.resolve()), 1))
    if not samples:
        raise RuntimeError(f"No HC_/PD_ wav under {audios_dir}")
    logger.info(f"Collected {len(samples)} samples from NeuroVoz audios")
    return samples


def collect_long_train_samples(data_root: str, binary: bool = True) -> List[Tuple[str, int]]:
    """NCMMSC: AD_dataset_long/train/{AD,HC[,MCI]}, binary: AD=0, HC=1."""
    root = Path(data_root)
    train_root = root / "AD_dataset_long" / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"Train root not found: {train_root}")
    samples: List[Tuple[str, int]] = []
    if binary:
        for d, label in [(train_root / "AD", 0), (train_root / "HC", 1)]:
            if d.exists():
                for wav in sorted(d.glob("*.wav")):
                    samples.append((str(wav.resolve()), label))
    else:
        for d, label in [(train_root / "AD", 0), (train_root / "MCI", 1), (train_root / "HC", 2)]:
            if d.exists():
                for wav in sorted(d.glob("*.wav")):
                    samples.append((str(wav.resolve()), label))
    if not samples:
        raise RuntimeError(f"No samples under {train_root}")
    logger.info(f"Collected {len(samples)} samples from NCMMSC long/train")
    return samples


def collect_train_samples(
    data_root: str,
    binary: bool = True,
) -> Tuple[List[Tuple[str, int]], int]:
    """根据 data_root 自动选择数据集，返回 (raw_samples, num_labels)。"""
    root = Path(data_root)
    if (root / "ADReSS-IS2020-data").exists():
        return collect_adress_train_samples(str(data_root)), 2
    if (root / "diagnosis" / "train" / "audio").exists():
        return collect_adresso21_train_samples(str(data_root)), 2
    if (root / "labels" / "train_split.csv").exists() and (root / "data").exists():
        return collect_edaic_train_samples(str(data_root)), 2
    if (root / "SW1-train").exists() and (root / "SW1-train" / "Metadata_train.csv").exists():
        return collect_speechwellness1_train_samples(str(data_root)), 2
    if (root / "train").exists() and (root / "training-groundtruth.csv").exists():
        return collect_adressm_train_samples(str(data_root)), 2
    if (root / "Reading-Task" / "audio" / "HC").exists() or (root / "Reading-Task" / "audio" / "PT").exists():
        return collect_androids_reading_train_samples(str(data_root)), 2
    if (root / "audios").exists() and (root / "metadata" / "metadata_hc.csv").exists():
        return collect_neurovoz_train_samples(str(data_root)), 2
    raw = collect_long_train_samples(str(data_root), binary=binary)
    return raw, 2 if binary else 3
