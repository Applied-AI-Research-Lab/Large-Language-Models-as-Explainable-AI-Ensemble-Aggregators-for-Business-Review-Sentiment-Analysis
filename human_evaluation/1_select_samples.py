#!/usr/bin/env python3
"""
T3.4 Human Evaluation Sample Selection
Selects 100 stratified samples for human annotation of LLaMA explanations.

Usage:
    python3 human_evaluation/1_select_samples.py

Output:
    human_evaluation/samples_for_annotation.csv - The 100 samples to annotate
    human_evaluation/annotation_template.csv - Empty template for annotators
    human_evaluation/sample_selection_report.json - Statistics on selection
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed: int = 42):
    random.seed(seed)


def categorize_sample(row: pd.Series) -> str:
    """
    Categorize each sample based on LLaMA behavior.
    
    Categories:
    - override: LLaMA disagrees with majority of base models
    - consensus: LLaMA agrees with majority
    - error: LLaMA prediction is wrong
    - edge: Ambiguous/conflicting base model predictions
    """
    true_label = row['stars'] - 1  # Convert 1-5 to 0-4
    llama_pred = row['llama_agg_pred_label']
    
    # Get base model predictions
    base_preds = [
        row.get('lr_pred_label'),
        row.get('svm_pred_label'),
        row.get('nb_pred_label'),
        row.get('bert_pred_label')
    ]
    base_preds = [p for p in base_preds if pd.notna(p)]
    
    if len(base_preds) == 0:
        return 'edge'
    
    # Check if LLaMA is correct
    llama_correct = (llama_pred == true_label)
    
    # Check base model agreement
    from collections import Counter
    pred_counts = Counter(base_preds)
    majority_pred, majority_count = pred_counts.most_common(1)[0]
    majority_agreement = majority_count / len(base_preds)
    
    # Categorize
    if not llama_correct:
        return 'error'
    elif majority_agreement < 0.5:
        return 'edge'
    elif llama_pred != majority_pred:
        return 'override'
    else:
        return 'consensus'


def select_stratified_samples(df: pd.DataFrame, n_total: int = 100) -> pd.DataFrame:
    """
    Select stratified samples across categories.
    
    Distribution:
    - override: 35 samples
    - consensus: 35 samples
    - error: 20 samples
    - edge: 10 samples
    """
    # Categorize all samples
    df['category'] = df.apply(categorize_sample, axis=1)
    
    # Define target counts (adjust based on availability)
    # Check actual availability first
    category_counts = df['category'].value_counts()
    
    target_counts = {
        'override': 35,
        'consensus': 35,
        'error': 25,  # Increased from 20 to compensate for limited edge cases
        'edge': min(10, category_counts.get('edge', 0))  # Take what's available
    }
    
    # Ensure we get 100 total by adjusting error category
    current_total = sum(target_counts.values())
    if current_total < 100:
        target_counts['error'] += (100 - current_total)
    
    selected_samples = []
    selection_stats = {}
    
    for category, target in target_counts.items():
        category_df = df[df['category'] == category]
        
        available = len(category_df)
        to_select = min(target, available)
        
        if to_select < target:
            print(f"Warning: Only {available} samples available for category '{category}' (requested {target})")
        
        # Random sample from this category
        selected = category_df.sample(n=to_select, random_state=42)
        selected_samples.append(selected)
        
        selection_stats[category] = {
            'requested': target,
            'available': available,
            'selected': to_select
        }
    
    # Combine and shuffle
    result = pd.concat(selected_samples).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add sample_id
    result.insert(0, 'sample_id', range(1, len(result) + 1))
    
    return result, selection_stats


def create_annotation_template(n_samples: int) -> pd.DataFrame:
    """Create empty template for annotators to fill in."""
    template = pd.DataFrame({
        'sample_id': range(1, n_samples + 1),
        'annotator_id': [''] * n_samples,
        'annotation_date': [''] * n_samples,
        
        # Ratings (1-5 scale)
        'faithfulness_rating': [''] * n_samples,  # Does explanation match base-model evidence?
        'usefulness_rating': [''] * n_samples,    # Would a business analyst find it actionable?
        'readability_rating': [''] * n_samples,   # Clarity of language
        
        # Optional comments
        'faithfulness_comment': [''] * n_samples,
        'usefulness_comment': [''] * n_samples,
        'readability_comment': [''] * n_samples,
        'general_notes': [''] * n_samples,
    })
    
    return template


def main():
    print("=" * 70)
    print("T3.4 Human Evaluation Sample Selection")
    print("=" * 70)
    
    # Load LLaMA predictions
    input_file = Path('Datasets/all_models_predictions_balanced_llama_reasoned.csv')
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"\nLoading predictions from {input_file}")
    df = pd.read_csv(input_file)
    print(f"Total samples available: {len(df)}")
    
    # Select stratified samples
    print("\nSelecting stratified samples...")
    selected_df, stats = select_stratified_samples(df, n_total=100)
    
    print(f"\nSelected {len(selected_df)} samples:")
    for category, info in stats.items():
        print(f"  {category}: {info['selected']}/{info['requested']} (available: {info['available']})")
    
    # Prepare output columns for annotators
    # Include all relevant information for context
    output_columns = [
        'sample_id',
        'review_id',
        'text',
        'stars',  # True label (1-5)
        'llama_agg_pred_stars',  # LLaMA prediction (1-5)
        'llama_agg_reasoning',  # The explanation to evaluate
        'category',  # override/consensus/error/edge
    ]
    
    # Add base model predictions for context (use stars, not labels, as these are what LLaMA sees)
    base_model_cols = [
        'lr_pred_stars', 'lr_pred_confidence',
        'svm_pred_stars', 'svm_pred_confidence',
        'nb_pred_stars', 'nb_pred_confidence',
        'bert_pred_stars', 'bert_pred_confidence',
    ]
    
    for col in base_model_cols:
        if col in selected_df.columns:
            output_columns.append(col)
    
    # Select only available columns
    available_cols = [c for c in output_columns if c in selected_df.columns]
    samples_for_annotation = selected_df[available_cols]
    
    # Save samples
    output_dir = Path('human_evaluation')
    output_dir.mkdir(exist_ok=True)
    
    samples_file = output_dir / 'samples_for_annotation.csv'
    samples_for_annotation.to_csv(samples_file, index=False)
    print(f"\n✓ Samples saved: {samples_file}")
    
    # Create annotation template
    template = create_annotation_template(len(selected_df))
    template_file = output_dir / 'annotation_template.csv'
    template.to_csv(template_file, index=False)
    print(f"✓ Annotation template: {template_file}")
    
    # Save selection report
    # Convert numpy types to Python native types for JSON serialization
    report = {
        'total_samples_available': int(len(df)),
        'samples_selected': int(len(selected_df)),
        'selection_by_category': {
            cat: {k: int(v) if isinstance(v, (int, np.integer)) else v 
                  for k, v in info.items()}
            for cat, info in stats.items()
        },
        'columns_included': list(available_cols),
    }
    
    report_file = output_dir / 'sample_selection_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Selection report: {report_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Sample Selection Complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Review samples: {samples_file}")
    print(f"  2. Print annotation guideline: human_evaluation/annotation_guideline.md")
    print(f"  3. Distribute to 2 annotators")
    print(f"  4. Collect ratings in: {template_file}")
    print(f"  5. Run analysis: python3 human_evaluation/2_analyze_annotations.py")


if __name__ == '__main__':
    main()
