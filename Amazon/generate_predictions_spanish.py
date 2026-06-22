#!/usr/bin/env python3
"""
Generate predictions from all trained models on Spanish Amazon test set.

Usage:
    python3 Amazon/generate_predictions_spanish.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_classical_model(model_path, vectorizer_path):
    """Load trained classical model and vectorizer."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_classical(model, vectorizer, texts):
    """Generate predictions from classical model."""
    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
    else:
        # For models without predict_proba, create one-hot
        probabilities = None
    
    return predictions, probabilities


def predict_bert(model_dir, texts, batch_size=32):
    """Generate predictions from BERT."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_logits = []
    all_preds = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, 
                          padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
            preds = torch.argmax(logits, dim=-1)
            all_logits.append(logits)
            all_preds.append(preds)
    
    logits = torch.cat(all_logits, dim=0).numpy()
    predictions = torch.cat(all_preds, dim=0).numpy()
    
    return predictions, logits


def main():
    print("Generating predictions for Spanish Amazon test set...")
    print("=" * 60)
    
    # Load test data
    test_df = pd.read_csv("Amazon/test_balanced.csv")
    texts = test_df['text'].astype(str).tolist()
    true_labels = test_df['stars'].values - 1  # Convert to 0-4
    
    results = {
        'review_id': test_df['review_id'],
        'text': test_df['text'],
        'true_label': true_labels,
    }
    
    # Classical models
    models_dir = Path("Amazon/models")
    
    for model_name in ['lr', 'svm', 'nb']:
        print(f"\nGenerating predictions from {model_name.upper()}...")
        model_path = models_dir / f"{model_name}_spanish.pkl"
        vectorizer_path = models_dir / "vectorizer_spanish.pkl"
        
        if not model_path.exists():
            print(f"  ⚠ Model not found: {model_path}")
            continue
        
        model, vectorizer = load_classical_model(model_path, vectorizer_path)
        preds, probs = predict_classical(model, vectorizer, texts)
        
        accuracy = accuracy_score(true_labels, preds)
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Store predictions in 0-4 format (labels) and 1-5 format (stars)
        results[f'{model_name}_pred_label'] = preds  # 0-4
        results[f'{model_name}_pred_stars'] = preds + 1  # 1-5
        
        # Calculate confidence as max probability
        if probs is not None:
            confidences = probs.max(axis=1)
            results[f'{model_name}_pred_confidence'] = confidences
            for i in range(5):
                results[f'{model_name}_prob_{i}'] = probs[:, i]
        else:
            results[f'{model_name}_pred_confidence'] = [None] * len(preds)
    
    # BERT
    print(f"\nGenerating predictions from BERT...")
    # Check both possible locations
    bert_dir = Path("Amazon/models/bert_spanish")
    if not bert_dir.exists():
        bert_dir = Path("Amazon/bert_spanish")
    if bert_dir.exists():
        preds, logits = predict_bert(bert_dir, texts)
        
        # Calculate probabilities from logits using softmax
        import numpy as np
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        confidences = probs.max(axis=1)
        
        accuracy = accuracy_score(true_labels, preds)
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Store predictions in 0-4 format (labels) and 1-5 format (stars)
        results['bert_pred_label'] = preds  # 0-4
        results['bert_pred_stars'] = preds + 1  # 1-5
        results['bert_pred_confidence'] = confidences
        
        for i in range(5):
            results[f'bert_logit_{i}'] = logits[:, i]
            results[f'bert_prob_{i}'] = probs[:, i]
    else:
        print(f"  ⚠ BERT model not found: {bert_dir}")
    
    # Save combined predictions
    print("\nSaving combined predictions...")
    results_df = pd.DataFrame(results)
    output_path = Path("Amazon/all_models_predictions_spanish.csv")
    results_df.to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")
    
    print("\n✓ Prediction generation complete!")
    print("\nNext steps:")
    print("  1. Run LLaMA standalone on Spanish reviews")
    print("  2. Run LLAMA_AGG with base model predictions")


if __name__ == "__main__":
    main()
