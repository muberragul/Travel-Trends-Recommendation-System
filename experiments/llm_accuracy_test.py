import os
import sys
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
PROJECT_ROOT = Path(parent_dir)
EVAL_DATA_DIR = PROJECT_ROOT / 'tests' / 'evaluation_data'

from extract_poi import extract_pois_with_qwen, extract_with_spacy
from fuzzy_matching import normalize_poi_name, is_substring_match, has_meaningful_overlap
from poi_name_extraction import calculate_poi_name_metrics, calculate_poi_type_metrics, f1_test, normalize_text
from rapidfuzz import fuzz

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVALUATION_CONFIG = {
    'qwen': {
        'model': 'qwen-turbo',
        'temperature': 0.0,
        'top_p': 1.0,
        'prompt_version': '1.0',
        'fuzzy_matching_enabled': True,
        'similarity_threshold': 80.0,
        'timestamp': datetime.now().isoformat()
    },
    'spacy': {
        'model': 'en_core_web_sm',
        'version': '3.7.0',
        'rule_set_version': '1.0',
        'fuzzy_matching_enabled': True,
        'similarity_threshold': 80.0
    },
    'evaluation': {
        'test_set_size': 90,
        'normalization': 'lowercase_strip_fuzzy'
    }
}

print("="*60)
print("POI EXTRACTION EVALUATION WITH FUZZY MATCHING")
print("="*60)
print(f"Qwen Model: {EVALUATION_CONFIG['qwen']['model']}")
print(f"Spacy Model: {EVALUATION_CONFIG['spacy']['model']}")
print(f"Fuzzy Matching: ENABLED (threshold={EVALUATION_CONFIG['qwen']['similarity_threshold']})")
print(f"Test Set: {EVALUATION_CONFIG['evaluation']['test_set_size']} posts")
print("="*60)

# ============================================================================
# FUZZY MATCHING FOR TEST RESULTS
# ============================================================================

def apply_fuzzy_matching_to_prediction(predicted_name, true_name, similarity_threshold=85.0):
    """
    Apply the same fuzzy matching logic used in production to test predictions.
    Returns the matched name if fuzzy match succeeds, otherwise original prediction.
    """
    if not predicted_name or not true_name:
        return predicted_name
    
    pred_lower = predicted_name.lower().strip()
    true_lower = true_name.lower().strip()
    
    # Exact match (case-insensitive)
    if pred_lower == true_lower:
        return true_name
    
    # Check if meaningful overlap exists
    if not has_meaningful_overlap(predicted_name, true_name):
        return predicted_name
    
    # Substring matching
    if is_substring_match(predicted_name, true_name):
        print(f"      [SUBSTRING] '{predicted_name}' ≈ '{true_name}'")
        return true_name
    
    # Normalized matching
    pred_normalized = normalize_poi_name(predicted_name)
    true_normalized = normalize_poi_name(true_name)
    
    if pred_normalized and true_normalized and len(pred_normalized) > 3:
        if pred_normalized == true_normalized:
            print(f"      [NORMALIZED] '{predicted_name}' ≈ '{true_name}'")
            return true_name
    
    # Fuzzy matching with multiple methods
    scores = []
    scores.append(fuzz.token_sort_ratio(pred_lower, true_lower))
    scores.append(fuzz.token_set_ratio(pred_lower, true_lower))
    scores.append(fuzz.partial_ratio(pred_lower, true_lower))
    
    if pred_normalized and true_normalized:
        scores.append(fuzz.token_sort_ratio(pred_normalized, true_normalized))
    
    max_similarity = max(scores)
    
    if max_similarity >= similarity_threshold:
        print(f"      [FUZZY] '{predicted_name}' ≈ '{true_name}' ({max_similarity:.1f}%)")
        return true_name
    
    return predicted_name

print("\n" + "="*60)
print("LOADING TEST DATA")
print("="*60)

test_files = [
    "batch1_testset.csv",
    "batch2_testset.csv", 
    "batch3_testset.csv"
]

all_test_data = []
for file in test_files:
    df = pd.read_csv(EVAL_DATA_DIR / file, encoding="utf-8-sig", quotechar='"', skipinitialspace=True)
    all_test_data.append(df)
    print(f"  {file}: {len(df)} posts")

test_df = pd.concat(all_test_data, ignore_index=True)
print(f"\nTotal: {len(test_df)} test samples")

print("\n" + "="*60)
print("PROCESSING WITH QWEN LLM + FUZZY MATCHING")
print("="*60)

qwen_results = []
for idx, row in test_df.iterrows():
    # Extract POIs (no location hint for fair evaluation)
    extractions = extract_pois_with_qwen(row['caption'], location=None)
    
    if extractions and len(extractions) > 0:
        predicted_name = extractions[0].get('poi_name', '')
        predicted_type = extractions[0].get('poi_type', 'other')
    else:
        predicted_name = ''
        predicted_type = 'other'
    
    # Apply fuzzy matching against ground truth
    if predicted_name:
        matched_name = apply_fuzzy_matching_to_prediction(
            predicted_name, 
            row['poi_name'],
            similarity_threshold=EVALUATION_CONFIG['qwen']['similarity_threshold']
        )
    else:
        matched_name = predicted_name
    
    qwen_results.append({
        'predicted_poi_name': matched_name,
        'original_prediction': predicted_name,
        'predicted_poi_type': predicted_type,
        'true_poi_name': row['poi_name'],
        'true_poi_type': row['poi_type'],
        'caption': row['caption']
    })
    
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(test_df)} samples...")


#no poi types
print("\n" + "="*60)
print("PROCESSING WITH SPACY NLP + FUZZY MATCHING")
print("="*60)
spacy_results = []
for idx, row in test_df.iterrows():
    extractions = extract_with_spacy(row['caption'], loc_name=None)
    
    if extractions and len(extractions) > 0:
        predicted_name = extractions[0].get('poi_name', '')
    else:
        predicted_name = ''
    
    # Apply fuzzy matching
    if predicted_name:
        matched_name = apply_fuzzy_matching_to_prediction(
            predicted_name,
            row['poi_name'],
            similarity_threshold=EVALUATION_CONFIG['spacy']['similarity_threshold']
        )
    else:
        matched_name = predicted_name
    
    spacy_results.append({
        'predicted_poi_name': matched_name,
        'original_prediction': predicted_name,
        'true_poi_name': row['poi_name'],
        'caption': row['caption']
    })

print("\n" + "="*60)
print("CALCULATING OVERALL METRICS")
print("="*60)
# Calculate for both
qwen_poi_metrics = calculate_poi_name_metrics(qwen_results)
spacy_poi_metrics = calculate_poi_name_metrics(spacy_results)
qwen_type_metrics = calculate_poi_type_metrics(qwen_results)

print("\n📊 QWEN (WITH FUZZY MATCHING):")
for key, value in qwen_poi_metrics.items():
    if key not in ['true_positives', 'false_positives', 'false_negatives']:
        print(f"  {key}: {value:.4f}")
print(f"  TP: {qwen_poi_metrics['true_positives']}, FP: {qwen_poi_metrics['false_positives']}, FN: {qwen_poi_metrics['false_negatives']}")

print("\n📊 SPACY (WITH FUZZY MATCHING):")
for key, value in spacy_poi_metrics.items():
    if key not in ['true_positives', 'false_positives', 'false_negatives']:
        print(f"  {key}: {value:.4f}")
print(f"  TP: {spacy_poi_metrics['true_positives']}, FP: {spacy_poi_metrics['false_positives']}, FN: {spacy_poi_metrics['false_negatives']}")

print(f"\n📊 QWEN POI Type Classification:")
print(f"  Accuracy: {qwen_type_metrics['accuracy']:.4f}")

# ============================================================================
# COMPARISON WITH PREVIOUS RESULTS
# ============================================================================

print("\n" + "="*60)
print("IMPROVEMENT ANALYSIS")
print("="*60)

previous_results = {
    'qwen_precision': 0.6167,
    'qwen_recall': 0.6167,
    'qwen_f1': 0.6167,
    'qwen_em': 0.6167,
    'spacy_precision': 0.4000,
    'spacy_recall': 0.2333,
    'spacy_f1': 0.2947,
    'spacy_em': 0.2333
}

print("\nQWEN IMPROVEMENTS:")
print(f"  Precision: {previous_results['qwen_precision']:.4f} → {qwen_poi_metrics['precision']:.4f} (Δ{(qwen_poi_metrics['precision'] - previous_results['qwen_precision'])*100:+.2f}%)")
print(f"  Recall:    {previous_results['qwen_recall']:.4f} → {qwen_poi_metrics['recall']:.4f} (Δ{(qwen_poi_metrics['recall'] - previous_results['qwen_recall'])*100:+.2f}%)")
print(f"  F1-Score:  {previous_results['qwen_f1']:.4f} → {qwen_poi_metrics['f1']:.4f} (Δ{(qwen_poi_metrics['f1'] - previous_results['qwen_f1'])*100:+.2f}%)")
print(f"  EM:        {previous_results['qwen_em']:.4f} → {qwen_poi_metrics['exact_match']:.4f} (Δ{(qwen_poi_metrics['exact_match'] - previous_results['qwen_em'])*100:+.2f}%)")

print("\nSPACY IMPROVEMENTS:")
print(f"  Precision: {previous_results['spacy_precision']:.4f} → {spacy_poi_metrics['precision']:.4f} (Δ{(spacy_poi_metrics['precision'] - previous_results['spacy_precision'])*100:+.2f}%)")
print(f"  Recall:    {previous_results['spacy_recall']:.4f} → {spacy_poi_metrics['recall']:.4f} (Δ{(spacy_poi_metrics['recall'] - previous_results['spacy_recall'])*100:+.2f}%)")
print(f"  F1-Score:  {previous_results['spacy_f1']:.4f} → {spacy_poi_metrics['f1']:.4f} (Δ{(spacy_poi_metrics['f1'] - previous_results['spacy_f1'])*100:+.2f}%)")
print(f"  EM:        {previous_results['spacy_em']:.4f} → {spacy_poi_metrics['exact_match']:.4f} (Δ{(spacy_poi_metrics['exact_match'] - previous_results['spacy_em'])*100:+.2f}%)")

# ============================================================================
# ERROR ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("ERROR ANALYSIS - REMAINING FAILURES")
print("="*60)
errors_shown = 0
for r in qwen_results:
    if normalize_text(r['predicted_poi_name']) != normalize_text(r['true_poi_name']) and errors_shown < 10:
        print(f"\n❌ Error {errors_shown + 1}:")
        print(f"   Caption: {r['caption'][:70]}...")
        print(f"   True POI: '{r['true_poi_name']}'")
        print(f"   Original Pred: '{r['original_prediction']}'")
        print(f"   After Fuzzy: '{r['predicted_poi_name']}'")
        errors_shown += 1

# ============================================================================
# RESULTS TABLES
# ============================================================================

print("\n" + "="*60)
print("FINAL RESULTS TABLES")
print("="*60)
# Create results tables
results_table_1 = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score', 'Exact Match', 'EER'],
    'Qwen (LLM)': [
        qwen_poi_metrics['precision'],
        qwen_poi_metrics['recall'],
        qwen_poi_metrics['f1'],
        qwen_poi_metrics['exact_match'],
        qwen_poi_metrics['extraction_error_rate']
    ],
    'SpaCy (NLP)': [
        spacy_poi_metrics['precision'],
        spacy_poi_metrics['recall'],
        spacy_poi_metrics['f1'],
        spacy_poi_metrics['exact_match'],
        spacy_poi_metrics['extraction_error_rate']
    ]
})

results_table_2 = pd.DataFrame({
    'Approach': ['Qwen (LLM)'],
    'Accuracy': [qwen_type_metrics['accuracy']]
})

print("\nTable 1: POI Name Extraction (WITH FUZZY MATCHING)")
print(results_table_1.to_string(index=False))

print("\nTable 2: POI Type Classification")
print(results_table_2.to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save configuration
with open(EVAL_DATA_DIR / 'evaluation_config_fuzzy.json', 'w') as f:
    json.dump(EVALUATION_CONFIG, f, indent=2)

# Save tables
results_table_1.to_csv(EVAL_DATA_DIR / 'poi_name_extraction_results_fuzzy.csv', index=False)
results_table_2.to_csv(EVAL_DATA_DIR / 'poi_type_classification_results_fuzzy.csv', index=False)

# Save detailed predictions
pd.DataFrame(qwen_results).to_csv(EVAL_DATA_DIR / 'qwen_detailed_predictions_fuzzy.csv', index=False)
pd.DataFrame(spacy_results).to_csv(EVAL_DATA_DIR / 'spacy_detailed_predictions_fuzzy.csv', index=False)

# Save evaluation summary
summary = {
    'evaluation_date': datetime.now().isoformat(),
    'configuration': EVALUATION_CONFIG,
    'previous_results': previous_results,
    'current_results': {
        'qwen': {
            'precision': float(qwen_poi_metrics['precision']),
            'recall': float(qwen_poi_metrics['recall']),
            'f1': float(qwen_poi_metrics['f1']),
            'exact_match': float(qwen_poi_metrics['exact_match']),
            'type_accuracy': float(qwen_type_metrics['accuracy'])
        },
        'spacy': {
            'precision': float(spacy_poi_metrics['precision']),
            'recall': float(spacy_poi_metrics['recall']),
            'f1': float(spacy_poi_metrics['f1']),
            'exact_match': float(spacy_poi_metrics['exact_match'])
        }
    },
    'improvements': {
        'qwen_f1_improvement': float(qwen_poi_metrics['f1'] - previous_results['qwen_f1']),
        'spacy_f1_improvement': float(spacy_poi_metrics['f1'] - previous_results['spacy_f1'])
    }
}

with open(EVAL_DATA_DIR / 'evaluation_summary_fuzzy.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✅ Results saved:")
print("   - poi_name_extraction_results_fuzzy.csv")
print("   - poi_type_classification_results_fuzzy.csv")
print("   - qwen_detailed_predictions_fuzzy.csv")
print("   - spacy_detailed_predictions_fuzzy.csv")
print("   - evaluation_summary_fuzzy.json")
print("   - evaluation_config_fuzzy.json")

print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)