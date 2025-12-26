import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. 加载现有的数据集
data = np.load('preprocessed_data/cmff_data.npz')
X_train = data['X_train']  # 仅包含正常样本
X_test_f = data['X_test_f']  # 仅包含故障样本

# 获取输入维度 (Window_Size, Features)
input_shape = (X_train.shape[1], X_train.shape[2])


# 2. 构建卷积自动编码器 (CAE)
def build_cae(input_shape):
    # 编码器：提取正常特征
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(2, padding='same')(x)

    # 解码器：尝试还原信号
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    # 调整输出长度以匹配输入 (处理奇偶窗口长度差异)
    decoded = layers.Conv1D(input_shape[1], 3, activation='linear', padding='same')(x)
    decoded = layers.Cropping1D(cropping=(0, x.shape[1] - input_shape[0]))(decoded)

    autoencoder = models.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


model = build_cae(input_shape)

# 3. 训练模型 (仅使用正常数据 X_train)
print("正在训练重构模型...")
history = model.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 4. 故障检测逻辑
# 计算重构误差
reconstruction_train = model.predict(X_train)
train_loss = np.mean(np.square(reconstruction_train - X_train), axis=(1, 2))

# 设定阈值 (例如训练集最大误差的 95 分位点)
threshold = np.percentile(train_loss, 95)
print(f"确定的检测阈值: {threshold:.4f}")

# 在故障测试集上验证
reconstruction_fault = model.predict(X_test_f)
fault_loss = np.mean(np.square(reconstruction_fault - X_test_f), axis=(1, 2))

# 计算分类准确率 (对于故障集，loss > threshold 即为正确分类)
detected_faults = np.sum(fault_loss > threshold)
accuracy = detected_faults / len(fault_loss)

print(f"\n测试分析结果:")
print(f"故障测试集样本数: {len(X_test_f)}")
print(f"成功检测出的故障数: {detected_faults}")
print(f"故障检出率 (Recall): {accuracy:.4f}")

# 5. 可视化
plt.hist(train_loss, bins=50, alpha=0.5, label='Normal (Train)')
plt.hist(fault_loss, bins=50, alpha=0.5, label='Fault (Test)')
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error Distribution')
plt.legend()
plt.show()