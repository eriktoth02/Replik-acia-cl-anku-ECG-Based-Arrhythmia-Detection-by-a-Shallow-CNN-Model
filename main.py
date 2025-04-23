import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils import load_record
from src.dataset import ECGDataset
from src.model import ShallowCNN
from src.train import train_model
from src.eval import evaluate_model

def main():
    record_id = '100'
    window_size = 100
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    desired_positive_count = 200  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Načítavam dáta...")
    beats, labels = load_record(record_id, window_size=window_size)

    X_train, X_temp, y_train, y_temp = train_test_split(beats, labels, test_size=0.3, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

    X_pos = X_train[y_train == 1]
    y_pos = y_train[y_train == 1]
    X_neg = X_train[y_train == 0]
    y_neg = y_train[y_train == 0]

    if len(X_pos) < desired_positive_count:
        repeat_factor = desired_positive_count // len(X_pos) + 1
        X_pos_oversampled = np.tile(X_pos, (repeat_factor, 1))[:desired_positive_count]
        y_pos_oversampled = np.ones(len(X_pos_oversampled))
    else:
        X_pos_oversampled = X_pos
        y_pos_oversampled = y_pos

    X_train_balanced = np.vstack((X_neg, X_pos_oversampled))
    y_train_balanced = np.concatenate((y_neg, y_pos_oversampled))

    indices = np.arange(len(X_train_balanced))
    np.random.shuffle(indices)
    X_train_balanced = X_train_balanced[indices]
    y_train_balanced = y_train_balanced[indices]

    print("Nový počet tried (po oversamplingu):", np.bincount(y_train_balanced.astype(int)))

    train_dataset = ECGDataset(X_train_balanced, y_train_balanced)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_length = 2 * window_size
    model = ShallowCNN(input_length=input_length)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    class_counts = np.bincount(y_train_balanced.astype(int))
    pos_weight_value = class_counts[0] / class_counts[1]
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Váha pre abnormálnu triedu (po oversamplingu): {pos_weight.item():.4f}")

    print("Trénujem model...")
    train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, device)

    torch.save(model.state_dict(), "outputs/saved_models/model_best.pt")
    print("✅ Model uložený do: model_best.pt")

    print("Vyhodnocujem na testovacích dátach...")
    evaluate_model(model, test_loader, device, threshold=0.3)

if __name__ == "__main__":
    main()
