#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024
# @Author  : Generated for custom mixer dataset
# @File    : prep_mixer.py

import os
import json
import shutil
from collections import Counter
import torch
import torchaudio
from torchaudio.transforms import Resample
import numpy as np

def trim_silence(waveform, top_db=40, frame_length=1024, hop_length=512):
    """Trim silence from audio waveform using RMS-based detection"""
    if waveform.shape[0] > 1:
        # Convert to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Calculate RMS in windows
    rms = []
    for i in range(0, max(1, len(waveform[0]) - frame_length + 1), hop_length):
        window = waveform[:, i:i+frame_length]
        rms_val = torch.sqrt(torch.mean(window**2))
        rms.append(rms_val.item())

    if not rms:
        return waveform

    rms = np.array(rms)
    ref = np.max(rms)

    # Find threshold
    threshold = ref * (10 ** (-top_db / 20))

    # Find non-silent frames
    non_silent = rms > threshold

    if not np.any(non_silent):
        return waveform

    # Find start and end indices
    start_frame = np.argmax(non_silent)
    end_frame = len(non_silent) - np.argmax(non_silent[::-1]) - 1

    # Convert to sample indices
    start_sample = start_frame * hop_length
    end_sample = min((end_frame + 1) * hop_length + frame_length, len(waveform[0]))

    return waveform[:, start_sample:end_sample]

def normalize_amplitude(waveform, target_amplitude=0.1):
    """Normalize waveform amplitude to target level"""
    if waveform.numel() == 0:
        return waveform

    current_max = torch.max(torch.abs(waveform))
    if current_max > 0:
        scale = target_amplitude / current_max
        waveform = waveform * scale
    return waveform

def pad_or_truncate(waveform, target_samples=480000):
    """Pad with zeros or truncate to target samples"""
    current_samples = waveform.shape[1]

    if current_samples < target_samples:
        # Pad with zeros at the end
        pad_samples = target_samples - current_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_samples))
    elif current_samples > target_samples:
        # Truncate
        waveform = waveform[:, :target_samples]

    return waveform

def preprocess_audio(input_path, output_path):
    """Preprocess single audio file"""
    # Load audio
    waveform, sample_rate = torchaudio.load(input_path)

    # Trim silence
    waveform = trim_silence(waveform, top_db=40)

    # Normalize amplitude
    waveform = normalize_amplitude(waveform, target_amplitude=0.1)

    # Resample to 16kHz
    if sample_rate != 16000:
        resampler = Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Pad/truncate to 30s (480000 samples at 16kHz)
    waveform = pad_or_truncate(waveform, target_samples=480000)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, waveform, 16000)

def prepare_mixer_dataset():
    """Prepare the mixer dataset for AST training with audio preprocessing"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dst_data_dir = os.path.join(script_dir, 'data')

    src_train = os.path.join(dst_data_dir, 'mixer_train_data.json')
    src_valid = os.path.join(dst_data_dir, 'mixer_eval_data.json')
    src_labels = os.path.join(dst_data_dir, 'class_labels_indices.csv')

    if not os.path.exists(src_train):
        alt_src_train = os.path.join(script_dir, '..', '..', 'datafiles', 'train_data.json')
        if os.path.exists(alt_src_train):
            src_train = alt_src_train
            src_valid = os.path.join(script_dir, '..', '..', 'datafiles', 'valid_data.json')
            src_labels = os.path.join(script_dir, '..', '..', 'datafiles', 'class_labels_indices.csv')
        else:
            print(f"ERROR: Source data not found at {src_train} or {alt_src_train}")
            print("Please ensure mixer_train_data.json and mixer_eval_data.json exist in egs/mixer/data/")
            return

    dst_train = os.path.join(dst_data_dir, 'mixer_train_data.json')
    dst_eval = os.path.join(dst_data_dir, 'mixer_eval_data.json')
    dst_labels = os.path.join(dst_data_dir, 'mixer_class_labels_indices.csv')

    processed_dir = os.path.join(dst_data_dir, 'processed_audio')
    dst_train_processed = os.path.join(dst_data_dir, 'mixer_train_data_processed.json')
    dst_eval_processed = os.path.join(dst_data_dir, 'mixer_eval_data_processed.json')

    os.makedirs(dst_data_dir, exist_ok=True)

    if not os.path.samefile(src_train, dst_train):
        print(f"Copying {src_train} to {dst_train}")
        shutil.copy2(src_train, dst_train)

    if not os.path.samefile(src_valid, dst_eval):
        print(f"Copying {src_valid} to {dst_eval}")
        shutil.copy2(src_valid, dst_eval)

    if not os.path.samefile(src_labels, dst_labels):
        print(f"Copying {src_labels} to {dst_labels}")
        shutil.copy2(src_labels, dst_labels)

    # Load and check the data
    with open(dst_train, 'r') as f:
        train_data = json.load(f)

    with open(dst_eval, 'r') as f:
        eval_data = json.load(f)

    print(f"Training samples: {len(train_data['data'])}")
    print(f"Evaluation samples: {len(eval_data['data'])}")

    # Count samples per class
    train_labels = [item['labels'] for item in train_data['data']]
    eval_labels = [item['labels'] for item in eval_data['data']]

    train_counts = Counter(train_labels)
    eval_counts = Counter(eval_labels)

    print("\nTraining set class distribution:")
    for cls, count in sorted(train_counts.items()):
        print(f"  {cls}: {count}")

    print("\nEvaluation set class distribution:")
    for cls, count in sorted(eval_counts.items()):
        print(f"  {cls}: {count}")

    # Get unique labels for subdirs
    all_labels = set(train_labels + eval_labels)

    # Create processed_audio subdirs
    for label in all_labels:
        os.makedirs(os.path.join(processed_dir, label), exist_ok=True)

    # Preprocess training data
    print("\nPreprocessing training data...")
    processed_train_data = {'data': []}
    for item in train_data['data']:
        original_path = item['wav']
        label = item['labels']
        filename = os.path.basename(original_path)
        processed_path = os.path.join(processed_dir, label, filename)

        preprocess_audio(original_path, processed_path)

        processed_item = item.copy()
        processed_item['wav'] = processed_path
        processed_train_data['data'].append(processed_item)

    # Preprocess evaluation data
    print("Preprocessing evaluation data...")
    processed_eval_data = {'data': []}
    for item in eval_data['data']:
        original_path = item['wav']
        label = item['labels']
        filename = os.path.basename(original_path)
        processed_path = os.path.join(processed_dir, label, filename)

        preprocess_audio(original_path, processed_path)

        processed_item = item.copy()
        processed_item['wav'] = processed_path
        processed_eval_data['data'].append(processed_item)

    # Save processed JSONs
    with open(dst_train_processed, 'w') as f:
        json.dump(processed_train_data, f, indent=2)

    with open(dst_eval_processed, 'w') as f:
        json.dump(processed_eval_data, f, indent=2)

    print(f"\nProcessed training data saved to {dst_train_processed}")
    print(f"Processed evaluation data saved to {dst_eval_processed}")
    print("All processed audio saved in processed_audio/ subdirectories")
    print("\nFinished mixer dataset preparation with preprocessing")

if __name__ == "__main__":
    prepare_mixer_dataset()