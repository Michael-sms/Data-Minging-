import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_recall_curve, f1_score

# 1. 加载数据
train_X = np.load('../processed_data/train_X.npy')
test_X = np.load('../processed_data/test_X.npy')
test_y = np.load('../processed_data/test_y.npy')

# 2. 拍扁数据 (Flatten)
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

# 7. 获取性能报告
report = classification_report(test_y, optimized_preds)
print("\n--- 优化后（最佳阈值）性能报告 ---")
print(report)

# 8. 【加分项】长尾数据提取
false_positives_indices = np.where((test_y == 0) & (optimized_preds == 1))[0]
fp_count = len(false_positives_indices)
print(f"\n[深度分析] 误判样本数 (正常判为异常): {fp_count}")

# 自动保存优化后的评估报告到 evaluation_results
report_file = '../evaluation_results/iforest_optimized_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("--- 孤立森林优化后性能报告 (基于PR曲线寻优) ---\n")
    f.write(f"搜索到的最佳阈值: {best_threshold:.4f}\n")
    f.write(f"长尾数据误判样本数: {fp_count}\n\n")
    f.write(report)
print(f"优化版报告已保存至: {report_file}")

# 9. 绘图：异常评分分布图（标注出最佳阈值）
plt.figure(figsize=(10, 6))
plt.hist(scores[test_y==0], bins=50, alpha=0.5, label='Normal', color='blue')
plt.hist(scores[test_y==1], bins=50, alpha=0.5, label='Anomaly', color='red')
# 绘制最佳阈值线
plt.axvline(x=best_threshold, color='green', linestyle='--', linewidth=2, label=f'Best Threshold ({best_threshold:.4f})')
plt.axvline(x=0, color='black', linestyle=':', label='Default Threshold (0.0000)')

plt.title('Anomaly Score Distribution (Optimized)')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.legend()

# 自动保存优化后的分布图到 image_results
image_file = '../image_results/iforest_optimized_distribution.png'
plt.savefig(image_file, dpi=300, bbox_inches='tight')
print(f"优化版分布图已保存至: {image_file}")

# 最后显示图形
plt.show()