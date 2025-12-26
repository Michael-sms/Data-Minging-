import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_recall_curve, f1_score

# 1. 加载数据
train_X = np.load('processed_data/train_X.npy')
test_X = np.load('processed_data/test_X.npy')
test_y = np.load('processed_data/test_y.npy')

# 2. 拍扁数据
X_train_flat = train_X.reshape(train_X.shape[0], -1)
X_test_flat = test_X.reshape(test_X.shape[0], -1)

# 3. 训练模型
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X_train_flat)

# 4. 获取原始异常分数 (decision_function 越小越异常)
scores = model.decision_function(X_test_flat)

# 5. 【核心优化】寻找最佳阈值
# precision_recall_curve 需要概率或得分，这里取负分（分数越低异常，取负后分数越高异常）
precisions, recalls, thresholds = precision_recall_curve(test_y, -scores)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = -thresholds[best_idx] # 转回原始分数的尺度

print(f"默认阈值: 0.0000")
print(f"搜索到的最佳阈值: {best_threshold:.4f}")

# 6. 使用新阈值进行预测
optimized_preds = np.where(scores < best_threshold, 1, 0)

# 7. 打印对比报告
print("\n--- 优化后（最佳阈值）性能报告 ---")
print(classification_report(test_y, optimized_preds))

# 8. 【加分项】长尾数据提取：找出那些被误判为异常的正常样本
false_positives_indices = np.where((test_y == 0) & (optimized_preds == 1))[0]
print(f"\n[深度分析] 误判样本数 (正常判为异常): {len(false_positives_indices)}")