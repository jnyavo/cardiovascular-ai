from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from keras import models

def display_confusion_matrix(y_pred: np.ndarray, X: np.ndarray, Y: np.ndarray, classes: list | np.ndarray = None) -> None:
    """
    Displays the confusion matrix for the model predictions.
    Parameters
    ----------
    y_pred: np.ndarray
        Model predictions.
    X: np.ndarray
        Input data.
    Y: np.ndarray
        True labels.
    """
    

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_int = np.argmax(Y, axis=1)

    cm = confusion_matrix(y_test_int, y_pred_classes)
    classes = classes if classes is not None else np.unique(y_test_int)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Matrice de confusion")
    fig.colorbar(cax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()