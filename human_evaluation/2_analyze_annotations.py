#!/usr/bin/env python3
"""
T3.4 Human Evaluation Analysis
Analyzes annotations from 2 human evaluators and computes inter-annotator agreement.

Usage:
    python3 human_evaluation/2_analyze_annotations.py \
        --annotator1 human_evaluation/annotations_annotator1.csv \
        --annotator2 human_evaluation/annotations_annotator2.csv

Output:
    human_evaluation/analysis_report.json - Statistical analysis
    human_evaluation/results_summary.md - Human-readable summary
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def cohens_kappa(rater1: pd.Series, rater2: pd.Series) -> float:
    """
    Calculate Cohen's Kappa for inter-annotator agreement.
    
    κ > 0.80: Almost perfect
    0.60-0.80: Substantial
    0.40-0.60: Moderate
    0.20-0.40: Fair
    < 0.20: Slight/poor
    """
    # Confusion matrix
    confusion = pd.crosstab(rater1, rater2)
    
    # Observed agreement
    observed = np.diag(confusion).sum() / confusion.sum().sum()
    
    # Expected agreement (by chance)
    rater1_marginals = confusion.sum(axis=1) / confusion.sum().sum()
    rater2_marginals = confusion.sum(axis=0) / confusion.sum().sum()
    expected = np.sum(rater1_marginals * rater2_marginals)
    
    # Cohen's Kappa
    if expected == 1:
        return 1.0
    
    kappa = (observed - expected) / (1 - expected)
    return kappa


def pearson_correlation(rater1: pd.Series, rater2: pd.Series) -> Tuple[float, float]:
    """Calculate Pearson correlation and p-value."""
    r, p = stats.pearsonr(rater1, rater2)
    return r, p


def analyze_dimension(df1: pd.DataFrame, df2: pd.DataFrame, dimension: str) -> Dict:
    """Analyze a single rating dimension."""
    ratings1 = df1[dimension]
    ratings2 = df2[dimension]
    
    # Convert to numeric (in case they're strings)
    ratings1 = pd.to_numeric(ratings1, errors='coerce')
    ratings2 = pd.to_numeric(ratings2, errors='coerce')
    
    # Remove any NaN values
    mask = ratings1.notna() & ratings2.notna()
    ratings1 = ratings1[mask]
    ratings2 = ratings2[mask]
    
    # Basic statistics
    mean1 = ratings1.mean()
    mean2 = ratings2.mean()
    std1 = ratings1.std()
    std2 = ratings2.std()
    
    # Inter-annotator agreement
    kappa = cohens_kappa(ratings1, ratings2)
    pearson_r, pearson_p = pearson_correlation(ratings1, ratings2)
    
    # Absolute agreement (exact match)
    exact_agreement = (ratings1 == ratings2).mean()
    
    # Within 1 point agreement
    within_1 = (abs(ratings1 - ratings2) <= 1).mean()
    
    return {
        'n_samples': len(ratings1),
        'annotator1_mean': float(mean1),
        'annotator2_mean': float(mean2),
        'annotator1_std': float(std1),
        'annotator2_std': float(std2),
        'cohens_kappa': float(kappa),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'exact_agreement': float(exact_agreement),
        'within_1_agreement': float(within_1),
    }


def analyze_by_category(df1: pd.DataFrame, df2: pd.DataFrame, dimension: str) -> Dict:
    """Analyze ratings broken down by sample category."""
    results = {}
    
    for category in df1['category'].unique():
        mask1 = df1['category'] == category
        mask2 = df2['category'] == category
        
        ratings1 = pd.to_numeric(df1.loc[mask1, dimension], errors='coerce')
        ratings2 = pd.to_numeric(df2.loc[mask2, dimension], errors='coerce')
        
        # Only analyze if we have matching samples
        if len(ratings1) == len(ratings2) and len(ratings1) > 0:
            mean_rating = ((ratings1 + ratings2) / 2).mean()
            results[category] = {
                'n_samples': len(ratings1),
                'mean_rating': float(mean_rating),
            }
    
    return results


def create_summary_report(results: Dict, output_dir: Path):
    """Create a human-readable summary report."""
    report_lines = [
        "# Human Evaluation Results Summary\n",
        "## T3.4: Human Evaluation of LLaMA Explanations\n",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n",
        f"**Samples:** {results['sample_size']}\n",
        f"**Annotators:** 2\n\n",
        "---\n\n",
        "## Inter-Annotator Agreement\n\n",
        "| Dimension | Cohen's κ | Interpretation | Pearson r | Exact Agreement | Within ±1 |\n",
        "|-----------|-----------|----------------|-----------|-----------------|-----------|\n",
    ]
    
    for dim, data in results['dimensions'].items():
        kappa = data['cohens_kappa']
        
        # Interpret kappa
        if kappa >= 0.80:
            interpretation = "Almost perfect"
        elif kappa >= 0.60:
            interpretation = "Substantial"
        elif kappa >= 0.40:
            interpretation = "Moderate"
        elif kappa >= 0.20:
            interpretation = "Fair"
        else:
            interpretation = "Slight"
        
        report_lines.append(
            f"| {dim.capitalize()} | {kappa:.3f} | {interpretation} | "
            f"{data['pearson_r']:.3f} | {data['exact_agreement']:.1%} | "
            f"{data['within_1_agreement']:.1%} |\n"
        )
    
    report_lines.extend([
        "\n## Mean Ratings by Dimension\n\n",
        "| Dimension | Annotator 1 | Annotator 2 | Overall Mean |\n",
        "|-----------|-------------|-------------|--------------|\n",
    ])
    
    for dim, data in results['dimensions'].items():
        overall_mean = (data['annotator1_mean'] + data['annotator2_mean']) / 2
        report_lines.append(
            f"| {dim.capitalize()} | {data['annotator1_mean']:.2f} ± {data['annotator1_std']:.2f} | "
            f"{data['annotator2_mean']:.2f} ± {data['annotator2_std']:.2f} | {overall_mean:.2f} |\n"
        )
    
    report_lines.extend([
        "\n## Ratings by Sample Category\n\n",
    ])
    
    for dim in results['by_category'].keys():
        report_lines.append(f"### {dim.capitalize()}\n\n")
        report_lines.append("| Category | N | Mean Rating |\n")
        report_lines.append("|----------|---|-------------|\n")
        
        for cat, data in results['by_category'][dim].items():
            report_lines.append(f"| {cat.capitalize()} | {data['n_samples']} | {data['mean_rating']:.2f} |\n")
        
        report_lines.append("\n")
    
    report_lines.extend([
        "---\n\n",
        "## Interpretation\n\n",
        "**Cohen's Kappa Interpretation:**\n",
        "- κ > 0.80: Almost perfect agreement\n",
        "- 0.60-0.80: Substantial agreement\n",
        "- 0.40-0.60: Moderate agreement\n",
        "- 0.20-0.40: Fair agreement\n",
        "- < 0.20: Slight/poor agreement\n\n",
        "**Key Findings:**\n",
    ])
    
    # Add key findings based on results
    best_kappa_dim = max(results['dimensions'].items(), key=lambda x: x[1]['cohens_kappa'])[0]
    worst_kappa_dim = min(results['dimensions'].items(), key=lambda x: x[1]['cohens_kappa'])[0]
    
    report_lines.append(f"- Highest inter-annotator agreement: {best_kappa_dim.capitalize()}\n")
    report_lines.append(f"- Lowest inter-annotator agreement: {worst_kappa_dim.capitalize()}\n")
    
    # Check if agreement is acceptable
    min_kappa = min(d['cohens_kappa'] for d in results['dimensions'].values())
    if min_kappa >= 0.60:
        report_lines.append("- All dimensions show substantial or better agreement (κ ≥ 0.60)\n")
    elif min_kappa >= 0.40:
        report_lines.append("- All dimensions show moderate or better agreement (κ ≥ 0.40)\n")
    else:
        report_lines.append("- Some dimensions show fair or poor agreement (κ < 0.40)\n")
    
    report_file = output_dir / 'results_summary.md'
    with open(report_file, 'w') as f:
        f.writelines(report_lines)
    
    print(f"✓ Summary report: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze human evaluation annotations")
    parser.add_argument("--annotator1", type=Path, required=True,
                        help="CSV file from annotator 1")
    parser.add_argument("--annotator2", type=Path, required=True,
                        help="CSV file from annotator 2")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("human_evaluation"),
                        help="Output directory")
    args = parser.parse_args()
    
    print("=" * 70)
    print("T3.4 Human Evaluation Analysis")
    print("=" * 70)
    
    # Load annotations
    print(f"\nLoading annotations...")
    df1 = pd.read_csv(args.annotator1)
    df2 = pd.read_csv(args.annotator2)
    
    print(f"Annotator 1: {len(df1)} samples")
    print(f"Annotator 2: {len(df2)} samples")
    
    # Ensure same samples
    if len(df1) != len(df2):
        print(f"Warning: Different number of samples. Using intersection.")
        common_ids = set(df1['sample_id']) & set(df2['sample_id'])
        df1 = df1[df1['sample_id'].isin(common_ids)].sort_values('sample_id')
        df2 = df2[df2['sample_id'].isin(common_ids)].sort_values('sample_id')
        print(f"Common samples: {len(df1)}")
    
    # Add category info if available
    samples_file = Path("human_evaluation/samples_for_annotation.csv")
    if samples_file.exists():
        samples_df = pd.read_csv(samples_file)
        if 'category' in samples_df.columns:
            df1 = df1.merge(samples_df[['sample_id', 'category']], on='sample_id', how='left')
            df2 = df2.merge(samples_df[['sample_id', 'category']], on='sample_id', how='left')
    
    # Analyze each dimension
    dimensions = ['faithfulness_rating', 'usefulness_rating', 'readability_rating']
    results = {
        'sample_size': len(df1),
        'dimensions': {},
        'by_category': {},
    }
    
    print("\nAnalyzing dimensions...")
    for dim in dimensions:
        print(f"\n{dim.replace('_', ' ').title()}:")
        dim_results = analyze_dimension(df1, df2, dim)
        results['dimensions'][dim.replace('_rating', '')] = dim_results
        
        print(f"  Cohen's κ: {dim_results['cohens_kappa']:.3f}")
        print(f"  Pearson r: {dim_results['pearson_r']:.3f} (p={dim_results['pearson_p']:.4f})")
        print(f"  Exact agreement: {dim_results['exact_agreement']:.1%}")
        print(f"  Mean ratings: {dim_results['annotator1_mean']:.2f} vs {dim_results['annotator2_mean']:.2f}")
        
        # Analyze by category
        if 'category' in df1.columns:
            results['by_category'][dim.replace('_rating', '')] = analyze_by_category(df1, df2, dim)
    
    # Save results
    args.output_dir.mkdir(exist_ok=True)
    
    results_file = args.output_dir / 'analysis_report.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Analysis report: {results_file}")
    
    # Create summary
    create_summary_report(results, args.output_dir)
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
