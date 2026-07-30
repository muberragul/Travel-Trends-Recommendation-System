from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sklearn.utils import resample

"""
def calculate_poi_name_metrics(results):
    
    # Prepare data
    y_true = []
    y_pred = []
    exact_matches = 0
    
    for r in results:
        true_name = normalize_text(r['true_poi_name'])
        pred_name = normalize_text(r['predicted_poi_name'])
        
        # Binary:  correct (1) or incorrect (0)
        if pred_name == true_name: 
            y_true.append(1)
            y_pred.append(1)
            exact_matches += 1
        elif pred_name == "":
            y_true.append(1)
            y_pred.append(0)  # False negative
        else:
            y_true.append(1)
            y_pred.append(0)  # False positive or wrong extraction
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    em = exact_matches / len(results)
    eer = 1 - em  # Extraction Error Rate
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'exact_match': em,
        'extraction_error_rate': eer
    }
"""
def calculate_poi_type_metrics(results):
    """Calculate Accuracy and Confusion Matrix for POI types"""
    
    y_true = [r['true_poi_type'] for r in results]
    y_pred = [r['predicted_poi_type'] for r in results]
    
    # Accuracy
    correct = sum([1 for t, p in zip(y_true, y_pred) if t == p])
    accuracy = correct / len(y_true)
    
    # Confusion matrix
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'labels': labels
    }

def normalize_text(text):
    """Normalize for comparison"""
    if text is None:
        return ""
    return str(text).lower().strip()

def calculate_poi_name_metrics(results):
    """Calculate Precision, Recall, F1, EM for POI names
    
    Precision: Of all POIs extracted, how many were correct?
    Recall: Of all true POIs, how many did we extract correctly?
    F1: Harmonic mean of precision and recall
    EM: Exact match rate
    """
    
    true_positives = 0   # Correct extractions
    false_positives = 0  # Wrong extractions (predicted something wrong)
    false_negatives = 0  # Missed extractions (predicted nothing or wrong)
    exact_matches = 0
    
    for r in results:
        true_name = normalize_text(r['true_poi_name'])
        pred_name = normalize_text(r['predicted_poi_name'])
        
        if pred_name == "" and true_name != "":
            # Model didn't extract anything, but should have
            false_negatives += 1
        elif pred_name == true_name and pred_name != "":
            # Correct extraction
            true_positives += 1
            exact_matches += 1
        elif pred_name != "" and pred_name != true_name:
            # Model extracted something wrong
            false_positives += 1
            false_negatives += 1  # Also a miss of the true POI
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    em = exact_matches / len(results) if len(results) > 0 else 0
    eer = 1 - em
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'exact_match': em,
        'extraction_error_rate': eer,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def calculate_poi_type_metrics(results):
    """Calculate Accuracy and Confusion Matrix for POI types"""
    
    y_true = [r['true_poi_type'] for r in results]
    y_pred = [r['predicted_poi_type'] for r in results]
    
    # Accuracy
    correct = sum([1 for t, p in zip(y_true, y_pred) if t == p])
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0
    
    # Confusion matrix
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'labels': labels
    }

def f1_test(results, n_iterations=1000):
    """Calculate 95% CI for F1 score using bootstrap"""
    f1_scores = []
    
    for i in range(n_iterations):
        # Resample with replacement
        sample = resample(results, n_samples=len(results), random_state=i)
        metrics = calculate_poi_name_metrics(sample)
        f1_scores.append(metrics['f1'])
    
    # Calculate 95% CI
    ci_lower = np.percentile(f1_scores, 2.5)
    ci_upper = np.percentile(f1_scores, 97.5)
    
    return ci_lower, ci_upper