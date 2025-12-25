import scipy.io as sio
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_mat_to_df(file_path):
    """自适应读取mat文件"""
    mat_data = sio.loadmat(file_path)
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    main_key = max(data_keys, key=lambda k: mat_data[k].size)
    raw_data = mat_data[main_key]
    return pd.DataFrame(raw_data)


def sliding_window(data, labels, size, s):
    """滑动窗口切片"""
    X, Y = [], []
    for i in range(0, len(data) - size, s):
        X.append(data[i: i + size])
        Y.append(1 if np.any(labels[i: i + size] == 1) else 0)
    return np.array(X), np.array(Y)


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


# --- 主逻辑保持 run_pipeline ---
def run_pipeline():
    # 1. 基础预处理
    normal_file = 'MultiphaseFlowFacilitydataset/CVACaseStudy/Training.mat'
    fault_files = [f'MultiphaseFlowFacilitydataset/CVACaseStudy/FaultyCase{i}.mat' for i in range(1, 7)]

    print("正在加载数据...")
    df_normal = load_mat_to_df(normal_file)
    df_normal['label'] = 0

    fault_list = [load_mat_to_df(f) for f in fault_files]
    for df in fault_list: df['label'] = 1
    df_all_fault = pd.concat(fault_list, axis=0)

    # 2. 筛选与归一化
    features = df_normal.columns[:-1]
    variances = df_normal[features].var()
    selected_features = variances[variances > 1e-6].index

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_normal[selected_features])
    fault_scaled = scaler.transform(df_all_fault[selected_features])

    # 3. 滑动窗口
    window_size, step = 60, 20
    X_normal_win, y_normal_win = sliding_window(train_scaled, df_normal['label'].values, window_size, step)
    X_fault_win, y_fault_win = sliding_window(fault_scaled, df_all_fault['label'].values, window_size, step)

    # 4. 数据集划分 (80%正常用于训练，20%正常+全部故障用于测试)
    X_train, X_test_normal, y_train, y_test_normal = train_test_split(
        X_normal_win, y_normal_win, test_size=0.2, random_state=42
    )
    X_test = np.concatenate([X_test_normal, X_fault_win], axis=0)
    y_test = np.concatenate([y_test_normal, y_fault_win], axis=0)

    # 5. 可视化并保存到 image_results
    inspect_and_save_visual(X_train, y_train, "Training_Set_Visual")
    inspect_and_save_visual(X_test, y_test, "Test_Set_Visual")

    # 6. 数据评估
    evaluate_processed_data(X_train, y_train, X_test, y_test)

    # 7. 保存数据文件
    save_dir = 'processed_data'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'train_X.npy'), X_train)
    np.save(os.path.join(save_dir, 'train_y.npy'), y_train)
    np.save(os.path.join(save_dir, 'test_X.npy'), X_test)
    np.save(os.path.join(save_dir, 'test_y.npy'), y_test)

    print("\n所有流程已完成！")


if __name__ == "__main__":
    run_pipeline()