import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, precision_recall_curve

# 1. 加载数据
train_X = np.load('processed_data/train_X.npy')
test_X = np.load('processed_data/test_X.npy')
test_y = np.load('processed_data/test_y.npy')

# 2. 拍扁数据 (Flatten)
X_train_flat = train_X.reshape(train_X.shape[0], -1)
X_test_flat = test_X.reshape(test_X.shape[0], -1)

# 3. 训练模型
# contamination 是预估的异常比例，可以根据 test_y 的实际比例微调
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X_train_flat)

# 4. 获取评分
# decision_function 分数越低越异常
scores = model.decision_function(X_test_flat)

# 5. 寻找最佳阈值 (这是一个高级分析点)
precision, recall, thresholds = precision_recall_curve(test_y, -scores)
f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]

# 6. 使用模型默认预测
preds = np.where(model.predict(X_test_flat) == -1, 1, 0)

# 7. 打印报告
print("--- Isolation Forest 性能报告 ---")
print(classification_report(test_y, preds))

# 8. 绘图：异常评分分布
plt.figure(figsize=(10, 6))
plt.hist(scores[test_y==0], bins=50, alpha=0.5, label='Normal', color='blue')
plt.hist(scores[test_y==1], bins=50, alpha=0.5, label='Anomaly', color='red')
plt.axvline(x=0, color='black', linestyle='--', label='Default Threshold')
plt.title('Anomaly Score Distribution')
plt.legend()
plt.show()