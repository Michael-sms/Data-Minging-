import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# --- 1. 定义 LSTM 自编码器 ---
class LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=1):
        super(LSTM_Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder: 压缩特征
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden2latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder: 还原特征
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        # --- Encode ---
        _, (h_n, _) = self.encoder(x)
        # 取最后一层的 hidden state
        h_n = h_n[-1]  # shape: (batch, hidden_dim)
        latent = self.hidden2latent(h_n)  # shape: (batch, latent_dim)

        # --- Decode ---
        # 将 latent 向量复制 seq_len 次，作为 decoder 的输入序列
        hidden = self.latent2hidden(latent)
        decoder_input = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        decoded, _ = self.decoder(decoder_input)
        reconstructed = self.output_layer(decoded)

        return reconstructed


# --- 2. 加载数据 ---
def load_data(data_dir='../processed_data'):
    print("正在加载数据...")
    X_train = np.load(f'{data_dir}/train_X.npy').astype(np.float32)
    # y_train 在训练中不需要，因为是无监督/自监督训练
    X_test = np.load(f'{data_dir}/test_X.npy').astype(np.float32)
    y_test = np.load(f'{data_dir}/test_y.npy').astype(np.int64)
    return X_train, X_test, y_test


# --- 3. 训练与阈值确定 ---
def train_and_evaluate():
    # 配置
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test, y_test = load_data()

    # 转换为 Tensor
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test)), batch_size=BATCH_SIZE)

    # 初始化模型
    input_dim = X_train.shape[2]  # 特征数
    model = LSTM_Autoencoder(input_dim).to(DEVICE)
    criterion = nn.MSELoss()  # 重构误差使用均方误差
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"开始在 {DEVICE} 上训练 Autoencoder (仅使用正常数据)...")
    loss_history = []

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)  # 目标是让输出尽可能接近输入
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Reconstruction Loss: {avg_loss:.6f}")

    # --- 阈值确定 (关键步骤) ---
    print("\n正在计算训练集重构误差分布...")
    model.eval()
    train_losses = []
    with torch.no_grad():
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            recon = model(x)
            # 计算每个样本的 loss (batch, seq, feat) -> (batch)
            loss = torch.mean((x - recon) ** 2, dim=[1, 2])
            train_losses.extend(loss.cpu().numpy())

    # 策略：使用训练集误差的 Mean + 3 * Std 作为阈值
    train_losses = np.array(train_losses)
    threshold = np.mean(train_losses) + 3 * np.std(train_losses)
    print(f"训练集平均误差: {np.mean(train_losses):.6f}, 标准差: {np.std(train_losses):.6f}")
    print(f"*** 设定异常判定阈值: {threshold:.6f} ***")

    # --- 测试集评估 ---
    print("\n正在评估测试集...")
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(DEVICE)
            recon = model(x)
            loss = torch.mean((x - recon) ** 2, dim=[1, 2])
            test_losses.extend(loss.cpu().numpy())

    test_losses = np.array(test_losses)

    # 如果 重构误差 > 阈值，则预测为 1 (异常)，否则为 0 (正常)
    y_pred = (test_losses > threshold).astype(int)

    # --- 结果可视化 ---
    print("\n=== 最终分类报告 ===")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fault']))


if __name__ == "__main__":
    train_and_evaluate()