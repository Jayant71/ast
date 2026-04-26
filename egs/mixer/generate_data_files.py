#!/usr/bin/env python3
import os
import json
import csv
import random
import argparse
import shutil
from collections import defaultdict

def generate_data_files(dataset_path, output_dir, train_ratio=0.8, seed=42, n_class=None):
    dataset_path = os.path.abspath(dataset_path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(dataset_path):
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    class_to_files = defaultdict(list)
    for entry in sorted(os.listdir(dataset_path)):
        entry_path = os.path.join(dataset_path, entry)
        if not os.path.isdir(entry_path):
            continue
        for fname in sorted(os.listdir(entry_path)):
            if fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                class_to_files[entry].append(os.path.join(entry_path, fname))

    if not class_to_files:
        print(f"ERROR: No audio files found in subdirectories of {dataset_path}")
        print("Expected structure: <dataset_path>/<class_label>/*.wav")
        return False

    classes = sorted(class_to_files.keys())

    if n_class is not None:
        classes = classes[:n_class]

    print(f"Found {len(classes)} classes: {classes}")
    for cls in classes:
        print(f"  {cls}: {len(class_to_files[cls])} files")

    labels_csv_path = os.path.join(output_dir, 'class_labels_indices.csv')
    with open(labels_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'mid', 'display_name'])
        for idx, cls in enumerate(classes):
            writer.writerow([idx, cls, cls])
    print(f"\nGenerated: {labels_csv_path}")

    mixer_labels_csv_path = os.path.join(output_dir, 'mixer_class_labels_indices.csv')
    shutil.copy2(labels_csv_path, mixer_labels_csv_path)

    train_data = []
    eval_data = []

    for cls in classes:
        files = class_to_files[cls][:]
        random.shuffle(files)
        split_idx = max(1, int(len(files) * train_ratio))
        for f in files[:split_idx]:
            train_data.append({"wav": f, "labels": cls})
        for f in files[split_idx:]:
            eval_data.append({"wav": f, "labels": cls})

    random.shuffle(train_data)
    random.shuffle(eval_data)

    train_json_path = os.path.join(output_dir, 'mixer_train_data.json')
    with open(train_json_path, 'w') as f:
        json.dump({"data": train_data}, f, indent=2)
    print(f"Generated: {train_json_path} ({len(train_data)} samples)")

    eval_json_path = os.path.join(output_dir, 'mixer_eval_data.json')
    with open(eval_json_path, 'w') as f:
        json.dump({"data": eval_data}, f, indent=2)
    print(f"Generated: {eval_json_path} ({len(eval_data)} samples)")

    train_proc_path = os.path.join(output_dir, 'mixer_train_data_processed.json')
    with open(train_proc_path, 'w') as f:
        json.dump({"data": train_data}, f, indent=2)
    print(f"Generated: {train_proc_path} ({len(train_data)} samples)")

    eval_proc_path = os.path.join(output_dir, 'mixer_eval_data_processed.json')
    with open(eval_proc_path, 'w') as f:
        json.dump({"data": eval_data}, f, indent=2)
    print(f"Generated: {eval_proc_path} ({len(eval_data)} samples)")

    print("\n--- Summary ---")
    print(f"Classes:       {len(classes)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples:  {len(eval_data)}")
    print(f"Train ratio:   {train_ratio}")
    print(f"\nAll files saved in: {output_dir}/")
    print(f"\nReady to train. Run:")
    print(f"  cd egs/mixer && ./run_mixer.sh")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all AST data files from a class-organized audio dataset")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset root with class subdirectories")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write generated files (default: egs/mixer/data/)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of data for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--n_class", type=int, default=None,
                        help="Limit number of classes (default: use all)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    generate_data_files(args.dataset_path, args.output_dir, args.train_ratio, args.seed, args.n_class)
