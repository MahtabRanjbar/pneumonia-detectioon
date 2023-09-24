import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)


def display_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Display a confusion matrix and save it as an image file if the save_path is not None.

    Args:
        y_true (list): List of true labels.
        y_pred (list): List of predicted labels.
        save_path (str or None): Path to save the confusion matrix image. If None, the image will not be saved.

    Returns:
        None
    """
    cf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(
        cf_matrix,
        annot=True,
        xticklabels=sorted(set(y_true)),
        yticklabels=sorted(set(y_true)),
        cbar=True,
    )
    plt.title("Confusion Matrix\n", fontsize=15)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=6)

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Confusion matrix saved as {save_path}")
    else:
        plt.tight_layout()
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()


def save_classification_report(y_true, y_pred, save_path=None):
    """
    Print the classification report and save it as a text file if the save_path is not None.

    Args:
        y_true (list): List of true labels.
        y_pred (list): List of predicted labels.
        save_path (str or None): Path to save the classification report. If None, the report will not be saved.

    Returns:
        None
    """
    report = classification_report(y_true, y_pred)

    print(report)

    if save_path is not None:
        with open(save_path, "w") as file:
            file.write(report)
        print(f"Classification report saved as {save_path}")
