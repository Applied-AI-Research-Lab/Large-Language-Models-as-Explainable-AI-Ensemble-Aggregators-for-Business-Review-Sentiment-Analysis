#!/usr/bin/env python3
"""
Prepare Spanish Amazon Reviews dataset (T3.5).

Samples 10,000 balanced reviews (2,000 per star rating),
splits into train/val/test, and saves to Amazon/ folder.

Usage:
    python3 Amazon/1_prepare_spanish_dataset.py
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42):
    random.seed(seed)


def load_balanced_samples(data_dir: Path, n_total: int = 10000, seed: int = 42):
    """
    Load balanced samples from Spanish Amazon dataset.
    Dataset has 40k per class (0-4), organized sequentially.
    """
    train_file = data_dir / "es" / "train.jsonl"
    
    samples_per_class = n_total // 5
    print(f"Loading {samples_per_class} samples per class (5 classes)...")
    
    data = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            label = i // 40000  # Determine class by position
            if label > 4:
                break
            
            if len(data[label]) < samples_per_class:
                item = json.loads(line)
                data[label].append({
                    'review_id': item['id'],
                    'text': item['text'],
                    'stars': item['label'] + 1,  # Convert 0-4 to 1-5
                })
            
            if all(len(v) >= samples_per_class for v in data.values()):
                break
    
    # Combine all classes
    all_data = []
    for label in range(5):
        all_data.extend(data[label])
    
    df = pd.DataFrame(all_data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df


def split_data(df: pd.DataFrame, train_frac=0.7, val_frac=0.15, seed=42):
    """Split into train/val/test."""
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Prepare Spanish Amazon dataset")
    parser.add_argument("--data-dir", type=Path, 
                       default=Path("Datasets/amazon_reviews_multi_es"))
    parser.add_argument("--output-dir", type=Path, default=Path("Amazon"))
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load balanced samples
    print(f"Loading {args.n_samples} balanced samples...")
    df = load_balanced_samples(args.data_dir, args.n_samples, args.seed)
    
    print(f"\nTotal loaded: {len(df)}")
    print("Star distribution:")
    print(df['stars'].value_counts().sort_index())
    
    # Split
    print("\nSplitting into train/val/test (70/15/15)...")
    train, val, test = split_data(df, seed=args.seed)
    
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")
    
    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    train.to_csv(args.output_dir / "train_balanced.csv", index=False)
    val.to_csv(args.output_dir / "validation_balanced.csv", index=False)
    test.to_csv(args.output_dir / "test_balanced.csv", index=False)
    
    print(f"\nSaved to {args.output_dir}/")
    print("  - train_balanced.csv")
    print("  - validation_balanced.csv")
    print("  - test_balanced.csv")


if __name__ == "__main__":
    main()
