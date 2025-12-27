import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

DATA_DIR = "../processed_data"
IMAGE_DIR = "../image_results"
EVAL_DIR = "../evaluation_results"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

class LSTM_AE(nn.Module):
    def __init__(self, n_features, hidden_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=n_features,
            batch_first=True
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        out, _ = self.decoder(h)
        return out


def reconstruction_error(x, x_hat):
    return torch.mean((x - x_hat) ** 2, dim=(1, 2)).cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_X = np.load(os.path.join(DATA_DIR, "train_X.npy"))
    test_X = np.load(os.path.join(DATA_DIR, "test_X.npy"))
    test_y = np.load(os.path.join(DATA_DIR, "test_y.npy"))

    seq_len = train_X.shape[1]
    n_features = train_X.shape[2]

    train_tensor = torch.tensor(train_X, dtype=torch.float32)
    test_tensor = torch.tensor(test_X, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=64,
        shuffle=True
    )

    model = LSTM_AE(
        n_features=n_features,
        hidden_dim=32,
        seq_len=seq_len
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    epochs = 30
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM-AE Training Loss")
    plt.savefig(os.path.join(IMAGE_DIR, "lstm_ae_training_loss.png"))
    plt.close()

    model.eval()
    with torch.no_grad():
        train_recon = model(train_tensor.to(device))
    train_errors = reconstruction_error(train_tensor, train_recon)

    threshold = np.percentile(train_errors, 95)

    with torch.no_grad():
        test_recon = model(test_tensor.to(device))
    test_errors = reconstruction_error(test_tensor, test_recon)

    test_pred = (test_errors > threshold).astype(int)

    acc = accuracy_score(test_y, test_pred)
    prec = precision_score(test_y, test_pred, zero_division=0)
    rec = recall_score(test_y, test_pred, zero_division=0)

    print("\n=== LSTM-AE Evaluation ===")
    print(f"Threshold: {threshold:.6f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")

    with open(os.path.join(EVAL_DIR, "lstm_ae_results.txt"), "w", encoding="utf-8") as f:
        f.write("LSTM-AE Unsupervised Anomaly Detection\n")
        f.write(f"Threshold (95%): {threshold:.6f}\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")

    plt.figure()
    plt.hist(test_errors, bins=50)
    plt.axvline(threshold, linestyle="--")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.title("Test Reconstruction Error Distribution")
    plt.savefig(os.path.join(IMAGE_DIR, "lstm_ae_error_distribution.png"))
    plt.close()


if __name__ == "__main__":
    main()
