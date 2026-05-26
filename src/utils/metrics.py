import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard evaluation metrics: Accuracy, Precision, Recall, and F1.
    Calculates both overall (Macro) metrics and checks for class-level performance.
    
    Args:
        y_true (np.ndarray): Ground truth class indices.
        y_pred (np.ndarray): Predicted class indices.
        
    Returns:
        Dict[str, float]: A dictionary containing calculated metrics.
    """
    # 1. Overall Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # 2. Precision, Recall, F1 (Macro Average)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # 3. Precision, Recall, F1 (Weighted Average)
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    return {
        "Accuracy": float(accuracy),
        "Precision_Macro": float(macro_prec),
        "Recall_Macro": float(macro_rec),
        "F1_Macro": float(macro_f1),
        "Precision_Weighted": float(weighted_prec),
        "Recall_Weighted": float(weighted_rec),
        "F1_Weighted": float(weighted_f1)
    }

def compute_class_performance(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Computes precision, recall, and F1-score for each individual class.
    Crucial for identifying underperforming minority classes and diagnosing class imbalance effects.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        class_names (List[str]): List of ordered class names.
        
    Returns:
        Dict[str, Dict[str, float]]: Per-class metrics dictionary.
    """
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )
    
    class_performance = {}
    for idx, name in enumerate(class_names):
        class_performance[name] = {
            "Precision": float(precisions[idx]),
            "Recall": float(recalls[idx]),
            "F1": float(f1s[idx]),
            "Support": int(supports[idx])
        }
        
    return class_performance

def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes standard confusion matrix."""
    return confusion_matrix(y_true, y_pred)
