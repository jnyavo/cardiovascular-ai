from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import seaborn as sns


def evaluate_classification(
    y_pred: np.ndarray, 
    X: np.ndarray, 
    Y: np.ndarray, 
    classes: Optional[Union[List[str], np.ndarray]] = None,
    normal_class_idx: int = 3,
    threshold: float = 0.5
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluates and displays comprehensive metrics for ECG classification including
    confusion matrix, precision, recall, F1 score, and ROC-AUC.
    
    Parameters
    ----------
    y_pred : np.ndarray
        Model predictions (probabilities) with shape (n_samples, n_classes).
    X : np.ndarray
        Input data (used for demonstration purposes when needed).
    Y : np.ndarray
        True labels (one-hot encoded) with shape (n_samples, n_classes).
    classes : Optional[Union[List[str], np.ndarray]], optional
        Class names for display purposes. If None, numerical indices will be used.
    normal_class_idx : int, optional
        Index of the "normal" class in the classification. Default is 3.
    threshold : float, optional
        Threshold for converting probabilities to binary predictions. Default is 0.5.
    
    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary containing evaluation metrics:
        - confusion_matrix: The confusion matrix
        - accuracy: Overall accuracy
        - precision: Precision for each class
        - recall: Recall for each class
        - f1: F1 score for each class
        - roc_auc: ROC-AUC score for each class
    """
    # Convert probabilities to class predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(Y, axis=1)
    
    # If classes are not provided, create numeric labels
    if classes is None:
        classes = [f"Class {i}" for i in range(Y.shape[1])]
    
    # Calculate various metrics
    metrics = {}
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    metrics["confusion_matrix"] = cm
    
    # Overall accuracy
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    metrics["accuracy"] = accuracy
    
    # Per-class metrics
    metrics["precision"] = precision_score(Y, y_pred_binary, average=None)
    metrics["recall"] = recall_score(Y, y_pred_binary, average=None)
    metrics["f1"] = f1_score(Y, y_pred_binary, average=None)
    
    # Calculate total samples per class for percentage calculations
    class_totals = np.sum(Y, axis=0)
    
    # ROC-AUC - handling multi-class
    metrics["roc_auc"] = []
    for i in range(Y.shape[1]):
        try:
            auc = roc_auc_score(Y[:, i], y_pred[:, i])
            metrics["roc_auc"].append(auc)
        except ValueError:
            # This happens when a class has only one label type
            metrics["roc_auc"].append(np.nan)
    metrics["roc_auc"] = np.array(metrics["roc_auc"])
    
    # Calculate false positives and false negatives specifically for normal class
    normal_idx = normal_class_idx
    
    # For normal class
    fp_normal = np.sum((y_pred_classes == normal_idx) & (y_true_classes != normal_idx))
    fn_normal = np.sum((y_pred_classes != normal_idx) & (y_true_classes == normal_idx))
    
    # Calculate total non-normal samples for percentage
    total_non_normal = np.sum(y_true_classes != normal_idx)
    total_normal = np.sum(y_true_classes == normal_idx)
    
    fp_normal_pct = (fp_normal / total_non_normal * 100) if total_non_normal > 0 else 0
    fn_normal_pct = (fn_normal / total_normal * 100) if total_normal > 0 else 0
    
    # For each class (including normal)
    fp_per_class = []
    fn_per_class = []
    fp_per_class_pct = []
    fn_per_class_pct = []
    
    for c in range(len(classes)):
        # False positives: predicted as class c but not actually class c
        fp = np.sum((y_pred_classes == c) & (y_true_classes != c))
        # False negatives: actually class c but predicted as something else
        fn = np.sum((y_pred_classes != c) & (y_true_classes == c))
        
        # Calculate the total number of non-class c samples for FP percentage
        total_non_c = np.sum(y_true_classes != c)
        # Calculate the total number of class c samples for FN percentage
        total_c = np.sum(y_true_classes == c)
        
        fp_pct = (fp / total_non_c * 100) if total_non_c > 0 else 0
        fn_pct = (fn / total_c * 100) if total_c > 0 else 0
        
        fp_per_class.append(fp)
        fn_per_class.append(fn)
        fp_per_class_pct.append(fp_pct)
        fn_per_class_pct.append(fn_pct)
    
    metrics["fp_per_class"] = np.array(fp_per_class)
    metrics["fn_per_class"] = np.array(fn_per_class)
    metrics["fp_per_class_pct"] = np.array(fp_per_class_pct)
    metrics["fn_per_class_pct"] = np.array(fn_per_class_pct)
    
    # Display normalized confusion matrix (percentages)
    plt.figure(figsize=(12, 10))
    
    # Create normalized confusion matrix (row-wise)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot regular confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Plot normalized confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized by Row)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()
    
    # Display metrics table with percentages
    metrics_table = np.zeros((len(classes), 7))
    for i in range(len(classes)):
        metrics_table[i, 0] = metrics["precision"][i]
        metrics_table[i, 1] = metrics["recall"][i]
        metrics_table[i, 2] = metrics["f1"][i]
        metrics_table[i, 3] = metrics["fp_per_class"][i]
        metrics_table[i, 4] = metrics["fp_per_class_pct"][i]
        metrics_table[i, 5] = metrics["fn_per_class"][i]
        metrics_table[i, 6] = metrics["fn_per_class_pct"][i]
    
    # Create a custom formatter for the annotation
    def custom_formatter(val, pos):
        if pos in [0, 1, 2]:  # Precision, Recall, F1
            return f'{val:.3f}'
        elif pos in [3, 5]:  # FP and FN counts
            return f'{int(val)}'
        else:  # FP and FN percentages
            return f'{val:.1f}%'
    
    plt.figure(figsize=(16, len(classes) * 0.8))
    
    # Create a mask to format each column differently
    fmt_array = np.empty_like(metrics_table, dtype=object)
    for i in range(metrics_table.shape[0]):
        for j in range(metrics_table.shape[1]):
            if j in [0, 1, 2]:  # Precision, Recall, F1
                fmt_array[i, j] = f'{metrics_table[i, j]:.3f}'
            elif j in [3, 5]:  # FP and FN counts
                fmt_array[i, j] = f'{int(metrics_table[i, j])}'
            else:  # FP and FN percentages
                fmt_array[i, j] = f'{metrics_table[i, j]:.1f}%'
    
    ax = sns.heatmap(metrics_table, annot=fmt_array, fmt='', cmap='YlGnBu',
                xticklabels=['Precision', 'Recall', 'F1', 'FP (count)', 'FP (%)', 'FN (count)', 'FN (%)'],
                yticklabels=classes)
    
    # Rotate the x labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.title('Classification Metrics per Class')
    plt.tight_layout()
    plt.show()
    
    # Display ROC-AUC scores
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(classes)), metrics["roc_auc"], color='skyblue')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.title('ROC-AUC Score per Class')
    plt.ylabel('ROC-AUC Score')
    plt.ylim(0, 1)
    
    # Highlight normal class
    if 0 <= normal_idx < len(classes):
        bars[normal_idx].set_color('orange')
    
    for i, v in enumerate(metrics["roc_auc"]):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary with special focus on normal vs abnormal classification
    print("\n===== Classification Summary =====")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro Avg ROC-AUC: {np.nanmean(metrics['roc_auc']):.4f}")
    print(f"Macro Avg F1-Score: {np.mean(metrics['f1']):.4f}")
    
    # Normal vs abnormal statistics
    print("\n===== Normal vs Abnormal ECG Detection =====")
    print(f"Normal ECG Precision: {metrics['precision'][normal_idx]:.4f}")
    print(f"Normal ECG Recall: {metrics['recall'][normal_idx]:.4f}")
    print(f"Normal ECG F1-Score: {metrics['f1'][normal_idx]:.4f}")
    print(f"False Positive Normal ECGs: {fp_normal} ({fp_normal_pct:.1f}% of abnormal samples)")
    print(f"False Negative Normal ECGs: {fn_normal} ({fn_normal_pct:.1f}% of normal samples)")
    
    return metrics


def analyze_misclassifications(
    y_pred: np.ndarray, 
    X: np.ndarray, 
    Y: np.ndarray, 
    classes: Optional[Union[List[str], np.ndarray]] = None,
    normal_class_idx: int = 3,
    max_examples: int = 5
) -> None:
    """
    Analyzes and displays examples of misclassifications with special focus on
    false positive and false negative normal ECGs.
    
    Parameters
    ----------
    y_pred : np.ndarray
        Model predictions (probabilities) with shape (n_samples, n_classes).
    X : np.ndarray
        Input ECG data.
    Y : np.ndarray
        True labels (one-hot encoded) with shape (n_samples, n_classes).
    classes : Optional[Union[List[str], np.ndarray]], optional
        Class names for display purposes. If None, numerical indices will be used.
    normal_class_idx : int, optional
        Index of the "normal" class in the classification. Default is 3.
    max_examples : int, optional
        Maximum number of examples to display for each misclassification type.
    """
    if classes is None:
        classes = [f"Class {i}" for i in range(Y.shape[1])]
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(Y, axis=1)
    
    # Find false positive normal ECGs (abnormal classified as normal)
    fp_normal_idx = np.where((y_pred_classes == normal_class_idx) & 
                             (y_true_classes != normal_class_idx))[0]
    
    # Find false negative normal ECGs (normal classified as abnormal)
    fn_normal_idx = np.where((y_pred_classes != normal_class_idx) & 
                             (y_true_classes == normal_class_idx))[0]
    
    # Calculate percentages
    total_abnormal = np.sum(y_true_classes != normal_class_idx)
    total_normal = np.sum(y_true_classes == normal_class_idx)
    
    fp_normal_pct = (len(fp_normal_idx) / total_abnormal * 100) if total_abnormal > 0 else 0
    fn_normal_pct = (len(fn_normal_idx) / total_normal * 100) if total_normal > 0 else 0
    
    # Display examples of false positive normal ECGs
    if len(fp_normal_idx) > 0:
        print(f"\n===== False Positive Normal ECGs (Abnormal classified as Normal) =====")
        print(f"Total: {len(fp_normal_idx)} ({fp_normal_pct:.1f}% of all abnormal samples)")
        for i, idx in enumerate(fp_normal_idx[:max_examples]):
            true_class = classes[y_true_classes[idx]]
            pred_class = classes[y_pred_classes[idx]]
            confidence = y_pred[idx, normal_class_idx]
            
            print(f"Example {i+1}:")
            print(f"  True class: {true_class}")
            print(f"  Predicted as: {pred_class} with confidence {confidence:.4f}")
            print(f"  All class probabilities: {', '.join([f'{classes[c]}: {p:.3f}' for c, p in enumerate(y_pred[idx])])}")
            
            # If display capability is needed, add code to plot the ECG here
    
    # Display examples of false negative normal ECGs
    if len(fn_normal_idx) > 0:
        print(f"\n===== False Negative Normal ECGs (Normal classified as Abnormal) =====")
        print(f"Total: {len(fn_normal_idx)} ({fn_normal_pct:.1f}% of all normal samples)")
        for i, idx in enumerate(fn_normal_idx[:max_examples]):
            true_class = classes[y_true_classes[idx]]
            pred_class = classes[y_pred_classes[idx]]
            normal_conf = y_pred[idx, normal_class_idx]
            pred_conf = y_pred[idx, y_pred_classes[idx]]
            
            print(f"Example {i+1}:")
            print(f"  True class: {true_class}")
            print(f"  Predicted as: {pred_class} with confidence {pred_conf:.4f}")
            print(f"  Normal class confidence: {normal_conf:.4f}")
            print(f"  All class probabilities: {', '.join([f'{classes[c]}: {p:.3f}' for c, p in enumerate(y_pred[idx])])}")
            
            # If display capability is needed, add code to plot the ECG here

