import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

def train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, device):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # VALIDÁCIA
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_preds_bin = [1 if p > 0.5 else 0 for p in val_preds]
        acc = accuracy_score(val_targets, val_preds_bin)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {np.mean(train_losses):.4f}, Val Acc: {acc:.4f}")
