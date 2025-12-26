import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def inspect_and_save_visual(X, y, title, save_dir='image_results'):
    """可视化并保存图片"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12, 4))
    sample_idx = np.random.randint(0, len(X))
    # 绘制前3个传感器作为示例
    for i in range(min(3, X.shape[2])):
        plt.plot(X[sample_idx, :, i], label=f'Sensor {i}')
    plt.title(f"{title} - Sample {sample_idx} (Label: {y[sample_idx]})")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_')}.png"))
    plt.close()  # 及时关闭避免内存占用
    print(f"可视化结果已保存至: {save_dir}")


def plot_correlation_heatmap(df, feature_names, save_dir='image_results'):
    """新增功能：绘制并保存传感器相关性热力图"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 关键修改：将 3D 窗口数据展平为 2D 矩阵，否则无法计算相关性
    # (样本数 * 60, 特征数)
    flattened_data = df.reshape(-1, df.shape[2])

    # 转换为 DataFrame 并加上特征名
    df = pd.DataFrame(flattened_data, columns=feature_names)

    # 计算相关系数矩阵
    corr_matrix = df.corr()
    plt.figure(figsize=(16, 12))
    # 使用 coolwarm 配色，中心点为0（白色）
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, linewidths=0.1)
    plt.title("Sensor Correlation Heatmap (Cranfield Multiphase Flow)", fontsize=16)
    save_path = os.path.join(save_dir, "sensor_correlation_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 高清保存
    plt.close()
    print(f"相关性热力图已保存至: {save_path}")


def evaluate_processed_data(X_train, y_train, X_test, y_test, save_dir='processed_data'):
    """
    数据评估脚本：分析处理后数据的质量并记录
    """
    report_path = os.path.join(save_dir, 'data_evaluation_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== CMFF 数据处理评估报告 ===\n\n")

        # 1. 样本分布评估
        train_counts = np.bincount(y_train.astype(int))
        test_counts = np.bincount(y_test.astype(int))
        f.write(f"[1. 样本分布]\n")
        f.write(f"训练集: 正常={train_counts[0]}, 异常={train_counts[1] if len(train_counts) > 1 else 0}\n")
        f.write(f"测试集: 正常={test_counts[0]}, 异常={test_counts[1]}\n")
        f.write(f"异常比例 (测试集): {test_counts[1] / len(y_test):.2%}\n\n")

        # 2. 数值质量评估 (检查归一化)
        f.write(f"[2. 数值质量 (训练集)]\n")
        f.write(f"全局均值: {np.mean(X_train):.4f} (理想值应接近0)\n")
        f.write(f"全局标准差: {np.std(X_train):.4f} (理想值应接近1)\n")
        f.write(f"最大值: {np.max(X_train):.4f}, 最小值: {np.min(X_train):.4f}\n\n")

        # 3. 窗口特征一致性
        f.write(f"[3. 结构检查]\n")
        f.write(f"窗口长度: {X_train.shape[1]}\n")
        f.write(f"特征维度: {X_train.shape[2]}\n")

    print(f"数据评估报告已生成: {report_path}")


def run_evaluation():
    # 1. 从文件夹加载处理好的数据
    data_dir = 'processed_data'
    print("正在加载数据进行质量检查...")
    X_train = np.load(f'{data_dir}/train_X.npy')
    y_train = np.load(f'{data_dir}/train_y.npy')
    X_test = np.load(f'{data_dir}/test_X.npy')
    y_test = np.load(f'{data_dir}/test_y.npy')
    selected_features = np.load(f'{data_dir}/selected_features.npy', allow_pickle=True)

    # 2. 执行你之前的评估流程
    plot_correlation_heatmap(X_train, selected_features)
    inspect_and_save_visual(X_train, y_train, "Training_Set_Visual")
    inspect_and_save_visual(X_test, y_test, "Test_Set_Visual")
    evaluate_processed_data(X_train, y_train, X_test, y_test)

    print("所有可视化图片已保存至 image_results 文件夹。")


if __name__ == "__main__":
    run_evaluation()