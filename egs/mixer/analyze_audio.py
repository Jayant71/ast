#!/usr/bin/env python3

import os
import json
import torchaudio
import torch
import numpy as np
from collections import defaultdict


def analyze_audio_dataset(json_path, output_summary=True):
    """Analyze audio dataset from JSON metadata"""
    with open(json_path, "r") as f:
        data = json.load(f)

    stats = defaultdict(list)
    total_files = len(data["data"])
    print(f"Analyzing {total_files} files from {json_path}")

    for i, item in enumerate(data["data"]):
        wav_path = item["wav"]
        label = item["labels"]

        if not os.path.exists(wav_path):
            print(f"Warning: {wav_path} not found")
            continue

        try:
            # Load audio
            y, sr = torchaudio.load(wav_path)
            y = y.squeeze(0).numpy()  # Convert to numpy

            # Duration
            duration = len(y) / sr
            stats["duration"].append(duration)

            # RMS energy
            rms = torchaudio.transforms.RMS()(torch.tensor(y).unsqueeze(0)).item()
            stats["rms"].append(20 * np.log10(rms) if rms > 0 else -np.inf)

            # Silence detection (segments below -40 dB)
            silence_threshold = 10 ** (-40 / 20)  # Convert dB to amplitude
            # Use sliding window for RMS
            hop_length = 512
            rms_frames = []
            for start in range(0, len(y) - hop_length, hop_length):
                frame = y[start : start + hop_length]
                rms_frame = np.sqrt(np.mean(frame**2))
                rms_frames.append(rms_frame)
            silence_frames = sum(1 for r in rms_frames if r < silence_threshold)
            silence_ratio = silence_frames / len(rms_frames) if rms_frames else 0
            stats["silence_ratio"].append(silence_ratio)

            # Per class
            stats[f"duration_{label}"].append(duration)
            stats[f"rms_{label}"].append(20 * np.log10(rms) if rms > 0 else -np.inf)
            stats[f"silence_{label}"].append(silence_ratio)

        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total_files}")

    if output_summary:
        print("\n=== Dataset Summary ===")
        print(f"Total files processed: {len(stats['duration'])}")
        durations = stats["duration"]
        print(
            f"Duration - Mean: {np.mean(durations):.2f}s, Std: {np.std(durations):.2f}s, Min: {np.min(durations):.2f}s, Max: {np.max(durations):.2f}s"
        )
        rms_vals = [r for r in stats["rms"] if r != -np.inf]
        print(
            f"RMS (dB) - Mean: {np.mean(rms_vals):.2f}, Std: {np.std(rms_vals):.2f}, Min: {np.min(rms_vals):.2f}, Max: {np.max(rms_vals):.2f}"
        )
        silence_ratios = stats["silence_ratio"]
        print(
            f"Silence Ratio - Mean: {np.mean(silence_ratios):.3f}, Std: {np.std(silence_ratios):.3f}, Min: {np.min(silence_ratios):.3f}, Max: {np.max(silence_ratios):.3f}"
        )

        # Per class summary
        labels = set(item["labels"] for item in data["data"])
        for label in sorted(labels):
            if f"duration_{label}" in stats:
                durs = stats[f"duration_{label}"]
                rms_l = [r for r in stats[f"rms_{label}"] if r != -np.inf]
                sils = stats[f"silence_{label}"]
                print(f"\n{label} (n={len(durs)}):")
                print(f"  Duration - Mean: {np.mean(durs):.2f}s")
                print(f"  RMS (dB) - Mean: {np.mean(rms_l):.2f}")
                print(f"  Silence Ratio - Mean: {np.mean(sils):.3f}")

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python analyze_audio.py <json_path>")
        sys.exit(1)

    json_path = sys.argv[1]
    analyze_audio_dataset(json_path)
