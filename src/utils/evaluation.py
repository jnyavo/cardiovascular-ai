from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import seaborn as sns
import os


def evaluate_classification(
    y_pred: np.ndarray, 
    X: np.ndarray, 
    Y: np.ndarray, 
    classes: Optional[Union[List[str], np.ndarray]] = None,
    normal_class_idx: Optional[int] = 3,
    threshold: float = 0.5,
    multi_label: bool = False
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
        True labels (one-hot encoded or multi-label) with shape (n_samples, n_classes).
    classes : Optional[Union[List[str], np.ndarray]], optional
        Class names for display purposes. If None, numerical indices will be used.
    normal_class_idx : Optional[int], optional
        Index of the "normal" class in the classification. Default is 3.
        If None, no specific normal/abnormal analysis will be performed.
    threshold : float, optional
        Threshold for converting probabilities to binary predictions. Default is 0.5.
    multi_label : bool, optional
        If True, treats the classification task as multi-label, where each sample can 
        belong to multiple classes. Default is False.
    
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
    
    # For overall accuracy metrics:
    if multi_label:
        # Multi-label metrics (example: consider classification correct if any true label is predicted)
        # Calculate multi-label accuracy - check if any of the true classes are predicted
        true_positive_samples = np.sum(np.logical_and(Y, y_pred_binary), axis=1) > 0
        accuracy = np.mean(true_positive_samples)
        
        # For confusion matrix visualization, we'll still need single-class representation
        # Use highest confidence prediction for visualization purposes
        y_pred_classes = np.argmax(y_pred, axis=1)
        # For multi-label, use the first true class for each sample for visualization
        y_true_classes = np.array([np.where(Y[i] > 0)[0][0] if np.sum(Y[i]) > 0 else 0 for i in range(Y.shape[0])])
    else:
        # Single-label classification (traditional)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(Y, axis=1)
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
    
    # If classes are not provided, create numeric labels
    if classes is None:
        classes = [f"Class {i}" for i in range(Y.shape[1])]
    
    # Calculate various metrics
    metrics = {}
    
    # Confusion matrix (still using the primary class predictions for visualization)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    metrics["confusion_matrix"] = cm
    
    # Overall accuracy (already calculated above based on multi_label parameter)
    metrics["accuracy"] = accuracy
    
    # Per-class metrics - these already work for multi-label
    metrics["precision"] = precision_score(Y, y_pred_binary, average=None, zero_division=0)
    metrics["recall"] = recall_score(Y, y_pred_binary, average=None, zero_division=0)
    metrics["f1"] = f1_score(Y, y_pred_binary, average=None, zero_division=0)
    
    # Add multi-label metrics
    if multi_label:
        metrics["precision_micro"] = precision_score(Y, y_pred_binary, average='micro', zero_division=0)
        metrics["recall_micro"] = recall_score(Y, y_pred_binary, average='micro', zero_division=0)
        metrics["f1_micro"] = f1_score(Y, y_pred_binary, average='micro', zero_division=0)
        metrics["precision_macro"] = precision_score(Y, y_pred_binary, average='macro', zero_division=0)
        metrics["recall_macro"] = recall_score(Y, y_pred_binary, average='macro', zero_division=0)
        metrics["f1_macro"] = f1_score(Y, y_pred_binary, average='macro', zero_division=0)
    
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
    if normal_class_idx is not None:
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
    
    # Highlight normal class if specified
    if normal_class_idx is not None and 0 <= normal_class_idx < len(classes):
        bars[normal_class_idx].set_color('orange')
    
    for i, v in enumerate(metrics["roc_auc"]):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()

    visualize_confidence_by_cluster(y_pred, Y, classes=classes)
    
    # Print summary with special focus on normal vs abnormal classification
    print("\n===== Classification Summary =====")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro Avg ROC-AUC: {np.nanmean(metrics['roc_auc']):.4f}")
    print(f"Macro Avg F1-Score: {np.mean(metrics['f1']):.4f}")
    
    # Normal vs abnormal statistics - only if normal_class_idx is provided
    if normal_class_idx is not None:
        print("\n===== Normal vs Abnormal ECG Detection =====")
        print(f"Normal ECG Precision: {metrics['precision'][normal_idx]:.4f}")
        print(f"Normal ECG Recall: {metrics['recall'][normal_idx]:.4f}")
        print(f"Normal ECG F1-Score: {metrics['f1'][normal_idx]:.4f}")
        print(f"False Positive Normal ECGs: {fp_normal} ({fp_normal_pct:.1f}% of abnormal samples)")
        print(f"False Negative Normal ECGs: {fn_normal} ({fn_normal_pct:.1f}% of normal samples)")
    
    visualize_false_prediction_distributions(y_pred, Y, classes=classes, multi_label=multi_label)
        
    
    return metrics


def visualize_confidence_by_cluster(
    y_pred: np.ndarray,
    Y: np.ndarray,
    classes: Optional[Union[List[str], np.ndarray]] = None,
    figsize: Tuple[int, int] = (12, 10),
    bins: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Visualizes the distribution of prediction confidence for false predictions,
    grouped by the predicted class (cluster).
    
    Parameters
    ----------
    y_pred : np.ndarray
        Model predictions (probabilities) with shape (n_samples, n_classes).
    Y : np.ndarray
        True labels (one-hot encoded) with shape (n_samples, n_classes).
    classes : Optional[Union[List[str], np.ndarray]], optional
        Class names for display purposes. If None, numerical indices will be used.
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches. Default is (12, 10).
    bins : int, optional
        Number of bins for the histogram. Default is 10.
    save_path : Optional[str], optional
        If provided, the figure will be saved to this path. Default is None.
    """
    # If classes are not provided, create numeric labels
    if classes is None:
        classes = [f"Class {i}" for i in range(Y.shape[1])]
    
    # Get predicted and true classes
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(Y, axis=1)
    
    # Get confidences (maximum probability for each prediction)
    confidences = np.max(y_pred, axis=1)
    
    # Find incorrect predictions
    incorrect_mask = y_pred_classes != y_true_classes
    
    # Create figure with a grid of subplots (one for each class)
    n_classes = len(classes)
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Summary statistics for false predictions
    total_false = np.sum(incorrect_mask)
    false_by_class = [np.sum((y_pred_classes == i) & incorrect_mask) for i in range(n_classes)]
    false_pct_by_class = [count / total_false * 100 if total_false > 0 else 0 for count in false_by_class]
    
    # Plot for each class
    for i in range(n_classes):
        # Filter to get only false predictions where the predicted class is i
        false_pred_mask = (y_pred_classes == i) & incorrect_mask
        class_confidences = confidences[false_pred_mask]
        
        ax = axes[i]
        if len(class_confidences) > 0:
            # Plot histogram of confidence values
            counts, edges, bars = ax.hist(
                class_confidences, 
                bins=bins, 
                range=(0, 1), 
                alpha=0.7, 
                color='salmon',
                edgecolor='black'
            )
            
            # Calculate mean confidence for false predictions
            mean_conf = np.mean(class_confidences) if len(class_confidences) > 0 else 0
            ax.axvline(mean_conf, color='red', linestyle='dashed', linewidth=2, 
                       label=f'Mean: {mean_conf:.2f}')
            
            # Add labels for each bar showing percentage
            bin_width = edges[1] - edges[0]
            total_count = len(class_confidences)
            
            for j, count in enumerate(counts):
                if count > 0:
                    percentage = count / total_count * 100
                    # Use a minimum offset and ensure it's proportional to data
                    vertical_offset = max(1, count * 0.08)  
                    ax.text(
                        edges[j] + bin_width/2, 
                        count + vertical_offset,
                        f'{percentage:.1f}%', 
                        ha='center', 
                        fontsize=8
                    )
            
            ax.set_title(f'{classes[i]}\n{false_by_class[i]} false preds ({false_pct_by_class[i]:.1f}% of all errors)')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Number of False Predictions')
            ax.legend()
            
            # Create bins for confidence ranges and count samples in each
            confidence_ranges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
            range_counts = []
            
            for lower, upper in confidence_ranges:
                range_count = np.sum((class_confidences >= lower) & (class_confidences < upper))
                range_pct = range_count / len(class_confidences) * 100 if len(class_confidences) > 0 else 0
                range_counts.append((range_count, range_pct))
            
            # Display counts for high confidence ranges as a table
            table_text = '\n'.join([
                f'{lower:.1f}-{upper:.1f}: {count} ({pct:.1f}%)' 
                for (lower, upper), (count, pct) in zip(confidence_ranges, range_counts)
            ])
            
            # Add a text box for confidence distribution
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.05, 0.95, table_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', bbox=props, linespacing=1.2)
        else:
            ax.text(0.5, 0.5, "No false predictions", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(classes[i])
        
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title and adjust layout
    plt.suptitle('Distribution of Confidence Scores for False Predictions by Predicted Class', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save figure if path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Create summary figure showing percentage of high-confidence errors by class
    plt.figure(figsize=(10, 6))
    
    # Count high confidence errors (>0.8) for each class
    high_conf_errors = []
    high_conf_error_pcts = []
    
    for i in range(n_classes):
        false_pred_mask = (y_pred_classes == i) & incorrect_mask
        class_confidences = confidences[false_pred_mask]
        
        high_conf_count = np.sum(class_confidences >= 0.8) if len(class_confidences) > 0 else 0
        high_conf_pct = high_conf_count / len(class_confidences) * 100 if len(class_confidences) > 0 else 0
        
        high_conf_errors.append(high_conf_count)
        high_conf_error_pcts.append(high_conf_pct)
    
    # Sort by percentage of high confidence errors
    sort_idx = np.argsort(high_conf_error_pcts)[::-1]
    sorted_classes = [classes[i] for i in sort_idx]
    sorted_pcts = [high_conf_error_pcts[i] for i in sort_idx]
    sorted_counts = [high_conf_errors[i] for i in sort_idx]
    
    # Plot bar chart
    bars = plt.bar(range(n_classes), sorted_pcts, color='salmon')
    
    # Add count labels
    for i, (count, pct) in enumerate(zip(sorted_counts, sorted_pcts)):
        if count > 0:
            plt.text(i, pct + 2, f'{count} ({pct:.1f}%)', ha='center')
    
    plt.xticks(range(n_classes), sorted_classes, rotation=45, ha='right')
    plt.ylabel('Percentage of False Predictions with Confidence ≥ 0.8')
    plt.title('High-Confidence False Predictions by Class')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path is not None:
        high_conf_path = save_path.replace('.', '_high_conf.')
        plt.savefig(high_conf_path, dpi=300, bbox_inches='tight')
        print(f"High confidence summary saved to {high_conf_path}")
    
    plt.show()



def analyze_misclassifications(
    y_pred: np.ndarray, 
    X: np.ndarray, 
    Y: np.ndarray, 
    classes: Optional[Union[List[str], np.ndarray]] = None,
    normal_class_idx: int = 3,
    max_examples: int = 5,
    multi_label: bool = False
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


def visualize_class_distribution(y, classes, figsize=(14, 8), color='skyblue', 
                                       top_n=None, save_path=None):
    """
    Visualizes the distribution of diagnostic classes in the PTB-XL dataset.
    
    This function takes the label matrix and class names, counts the number of records 
    for each diagnostic class, and creates a bar chart to visualize this distribution.
    
    Parameters:
    -----------
    y : numpy.ndarray
        Binary label matrix where rows are samples and columns are classes (one-hot encoded or multi-label).
    classes : list or numpy.ndarray
        Names of the classes corresponding to the columns in y.
    figsize : tuple, optional
        Figure size for the plot. Default is (14, 8).
    color : str or list, optional
        Color(s) for the bars. Default is 'skyblue'. Can be a single color or a list of colors.
    top_n : int, optional
        If provided, only the top N most frequent classes are shown. Default is None (all classes).
    save_path : str, optional
        If provided, the figure will be saved to this path. Default is None (not saved).
        
    Returns:
    --------
    tuple
        (sorted_counts, sorted_classes, figure) - Counts of each class, class names (sorted by frequency), 
        and the matplotlib figure object.
    
    """
    
    # Count instances per class (sum of binary matrix)
    class_counts = np.sum(y, axis=0)
    
    # Sort by frequency for better visualization
    sorted_indices = np.argsort(class_counts)[::-1]
    sorted_classes = np.array(classes)[sorted_indices]
    sorted_counts = class_counts[sorted_indices]
    
    # Limit to top N classes if specified
    if top_n is not None and top_n < len(sorted_classes):
        sorted_classes = sorted_classes[:top_n]
        sorted_counts = sorted_counts[:top_n]
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle color parameter
    if isinstance(color, list) and len(color) == len(sorted_classes):
        bars = ax.bar(range(len(sorted_classes)), sorted_counts, color=color)
    else:
        bars = ax.bar(range(len(sorted_classes)), sorted_counts, color=color)
    
    # Add value labels above each bar with more padding
    for i, count in enumerate(sorted_counts):
        ax.text(i, count + max(sorted_counts)*0.03, f'{int(count)}', ha='center', fontsize=11)
    
    # Add labels and title
    ax.set_xticks(range(len(sorted_classes)))
    ax.set_xticklabels(sorted_classes, rotation=45, ha='right')
    ax.set_title('Diagnostic Class Distribution', fontsize=16)
    ax.set_xlabel('Diagnostic Classes', fontsize=14)
    ax.set_ylabel('Number of Instances', fontsize=14)
    
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add information about the dataset as a text box
    total_records = len(y)
    multi_label_count = np.sum(np.sum(y, axis=1) > 1)
    multi_label_ratio = multi_label_count / len(y) if len(y) > 0 else 0
    
    # Create a text box for dataset information
    info_text = f'Dataset Summary:\n' \
                f'Total records: {total_records}\n' \
                f'Multi-label records: {multi_label_count} ({multi_label_ratio:.1%})'
    
    # Add a text box in the upper right corner
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.97, 0.97, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, linespacing=1.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    
    return sorted_counts, sorted_classes, fig



def get_false_prediction_indices(
    y_pred: np.ndarray,
    Y: np.ndarray,
    multi_label: bool = False,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Returns indices of all false predictions in the dataset.
    
    Parameters
    ----------
    y_pred : np.ndarray
        Model predictions (probabilities) with shape (n_samples, n_classes).
    Y : np.ndarray
        True labels (one-hot encoded or multi-label) with shape (n_samples, n_classes).
    multi_label : bool, optional
        If True, treats the classification task as multi-label. Default is False.
    threshold : float, optional
        Threshold for converting probabilities to binary predictions. Default is 0.5.
    
    Returns
    -------
    np.ndarray
        Array of indices where the prediction was incorrect.
    """
    # Convert probabilities to class predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    if multi_label:
        # For multi-label: consider a prediction false if the predicted class 
        # with highest confidence is not in the true labels
        y_pred_classes = np.argmax(y_pred, axis=1)
        false_indices = np.where(~np.array([y_pred_classes[i] in np.where(Y[i] > 0)[0]  
                                           for i in range(len(Y))]))[0]
    else:
        # For single-label: simply compare the predicted and true class
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(Y, axis=1)
        false_indices = np.where(y_pred_classes != y_true_classes)[0]
    
    return false_indices


def visualize_false_prediction_distributions(
    y_pred: np.ndarray,
    Y: np.ndarray,
    classes: Optional[Union[List[str], np.ndarray]] = None,
    multi_label: bool = False,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Analyzes false predictions and visualizes the average prediction values for each class.
    
    Parameters
    ----------
    y_pred : np.ndarray
        Model predictions (probabilities) with shape (n_samples, n_classes).
    Y : np.ndarray
        True labels (one-hot encoded) with shape (n_samples, n_classes).
    classes : Optional[Union[List[str], np.ndarray]], optional
        Class names for display purposes. If None, numerical indices will be used.
    multi_label : bool, optional
        If True, treats the classification task as multi-label. Default is False.
    threshold : float, optional
        Threshold for converting probabilities to binary predictions. Default is 0.5.
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches. Default is (14, 10).
    save_path : Optional[str], optional
        If provided, the figure will be saved to this path. Default is None.
    """
    # If classes are not provided, create numeric labels
    if classes is None:
        classes = [f"Class {i}" for i in range(Y.shape[1])]
    
    n_classes = len(classes)
    
    # Get indices of false predictions
    false_indices = get_false_prediction_indices(y_pred, Y, multi_label, threshold)
    
    if len(false_indices) == 0:
        print("No false predictions found!")
        return
    
    # Get the false predictions and their true labels
    false_preds = y_pred[false_indices]
    false_Y = Y[false_indices]
    
    # Get true and predicted classes for false predictions
    if multi_label:
        # For multi-label: use the first positive label for visualization purposes
        y_true_classes = np.array([np.where(false_Y[i] > 0)[0][0] 
                                  if np.sum(false_Y[i]) > 0 else 0 
                                  for i in range(false_Y.shape[0])])
    else:
        y_true_classes = np.argmax(false_Y, axis=1)
    
    # Create figure for heatmap of average predictions by true class
    plt.figure(figsize=figsize)
    
    # Calculate average prediction per class for each true class
    avg_preds_by_true_class = np.zeros((n_classes, n_classes))
    class_counts = np.zeros(n_classes)
    
    # Calculate average prediction values for each true class
    for i in range(n_classes):
        # Get indices where true class is i
        class_mask = y_true_classes == i
        class_count = np.sum(class_mask)
        class_counts[i] = class_count
        
        if class_count > 0:
            # Calculate average prediction values for this class
            avg_preds_by_true_class[i] = np.mean(false_preds[class_mask], axis=0)
    
    # Filter out classes with no false predictions
    non_empty_mask = class_counts > 0
    filtered_classes = [classes[i] for i in range(n_classes) if non_empty_mask[i]]
    filtered_avg_preds = avg_preds_by_true_class[non_empty_mask]
    filtered_counts = class_counts[non_empty_mask]
    
    if len(filtered_classes) == 0:
        print("No classes with false predictions to visualize")
        return
    
    # Create heatmap
    sns.heatmap(
        filtered_avg_preds,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=classes,
        yticklabels=[f"{c} (n={int(count)})" for c, count in zip(filtered_classes, filtered_counts)]
    )
    
    plt.xlabel('Predicted Class Probabilities')
    plt.ylabel('True Class (with false prediction count)')
    plt.title('Average Prediction Values for False Predictions by True Class')
    
    # Add colorbar label
    cbar = plt.gca().collections[0].colorbar
    cbar.set_label('Average Prediction Value')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Create a second visualization showing the confusion pattern
    plt.figure(figsize=(figsize[0], figsize[1]/1.5))
    
    # For each true class, find the highest predicted class (excluding the true class)
    strongest_confusion = []
    confusion_values = []
    confusion_labels = []
    
    for i in range(len(filtered_classes)):
        true_class_idx = np.where(classes == filtered_classes[i])[0][0]
        avg_preds = filtered_avg_preds[i].copy()
        
        # Set the value for the true class to -1 to find the max of other classes
        avg_preds[true_class_idx] = -1
        
        # Find the highest predicted class (which is not the true class)
        highest_pred_idx = np.argmax(avg_preds)
        highest_pred_value = filtered_avg_preds[i, highest_pred_idx]
        
        confusion_labels.append(f"{filtered_classes[i]} → {classes[highest_pred_idx]}")
        confusion_values.append(highest_pred_value)
        strongest_confusion.append((filtered_classes[i], classes[highest_pred_idx], highest_pred_value))
    
    # Sort by confusion strength
    sort_idx = np.argsort(confusion_values)[::-1]
    sorted_labels = [confusion_labels[i] for i in sort_idx]
    sorted_values = [confusion_values[i] for i in sort_idx]
    sorted_counts = [filtered_counts[i] for i in sort_idx]
    
    # Plot bar chart of confusion patterns
    bars = plt.bar(range(len(sorted_labels)), sorted_values, color='salmon')
    
    # Add count labels
    for i, (count, value) in enumerate(zip(sorted_counts, sorted_values)):
        plt.text(i, value + 0.02, f'n={int(count)}', ha='center')
    
    plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=45, ha='right')
    plt.ylabel('Average Prediction Strength')
    plt.title('Strongest Class Confusion Patterns')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, min(1.0, max(sorted_values) + 0.15))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        confusion_path = save_path.replace('.', '_confusion_patterns.')
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        print(f"Confusion patterns saved to {confusion_path}")
    
    plt.show()