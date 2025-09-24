import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

def evaluate_model(model, dataloader, class_names, device, save_path=None):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- Metrics ---
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    print(f"\n Accuracy: {acc * 100:.2f}%")
    print("\n Classification Report:")
    print(report)

    # --- Confusion matrix plot ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (%)")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f" Confusion matrix saved to: {save_path}")
        plt.close()

    return acc, cm, report
