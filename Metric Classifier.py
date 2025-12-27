import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix


# --- 1. Encoder Design ---
class MetricEncoder(nn.Module):
    def __init__(self, input_dim):
        super(MetricEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        # 映射到特征球空间
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        feat = self.conv(x).flatten(1)
        return self.fc(feat)


# --- 2. Adaptive Inference Engine ---
class AdaptiveExperiment:
    def __init__(self, input_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MetricEncoder(input_dim).to(self.device)

    def run_training(self, X_train):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float()), batch_size=64, shuffle=True)

        self.model.train()
        for epoch in range(20):
            for [bx] in loader:
                optimizer.zero_grad()
                z = self.model(bx.to(self.device))
                # Center Loss: 强制正常样本聚类在原点周围
                loss = torch.mean(torch.pow(torch.norm(z, p=2, dim=1), 2))
                loss.backward()
                optimizer.step()

    def evaluate_adaptive(self, X_train, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            # 建立基准
            z_train = self.model(torch.from_numpy(X_train).float().to(self.device))
            z_test = self.model(torch.from_numpy(X_test).float().to(self.device))

            # 计算欧氏距离
            dists = torch.norm(z_test, p=2, dim=1).cpu().numpy()
            train_dists = torch.norm(z_train, p=2, dim=1).cpu().numpy()

            # 动态阈值优化：取训练集分布的 90 分位数作为更严谨的边界
            # 在学术上，这被称为 Sensitivity-Specific Trade-off
            threshold = np.percentile(train_dists, 90)
            y_pred = (dists > threshold).astype(int)

        print("\n" + "=" * 60)
        print("ADAPTIVE METRIC CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(y_test, y_pred, digits=4))

        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\nTN: {cm[0, 0]} | FP: {cm[0, 1]}\nFN: {cm[1, 0]} | TP: {cm[1, 1]}")


# --- 3. Run ---
if __name__ == "__main__":
    X_train = np.load('processed_data/train_X.npy')
    X_test = np.load('processed_data/test_X.npy')
    y_test = np.load('processed_data/test_y.npy')

    exp = AdaptiveExperiment(X_train.shape[2])
    exp.run_training(X_train)
    # 使用分位数自适应阈值
    exp.evaluate_adaptive(X_train, X_test, y_test)