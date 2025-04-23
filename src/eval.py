import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, data_loader, device, threshold=0.5):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)  
            preds.extend(outputs.cpu().numpy())
            targets.extend(labels.numpy())

    print("\n🔍 Prvých 20 predikcií (pravdepodobnosti):")
    print(np.round(preds[:20], 3))

    preds_bin = [1 if p > threshold else 0 for p in preds]

    acc = accuracy_score(targets, preds_bin)
    prec = precision_score(targets, preds_bin, zero_division=0)
    rec = recall_score(targets, preds_bin, zero_division=0)
    f1 = f1_score(targets, preds_bin, zero_division=0)
    auc = roc_auc_score(targets, preds)
    
    tn, fp, fn, tp = confusion_matrix(targets, preds_bin).ravel()
    specificity = tn / (tn + fp)

    print("\n🔍 Evaluation Metrics:")
    print(f"Threshold    : {threshold}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"AUC          : {auc:.4f}")

    fpr, tpr, _ = roc_curve(targets, preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
