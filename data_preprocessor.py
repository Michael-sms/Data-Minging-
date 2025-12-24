import scipy.io as sio
import pandas as pd
import scipy.io as sio
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt  # 用于查看数据形态
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # 用于科学划分数据集


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
        # 异常检测逻辑：窗口内包含任何故障点即标记为1
        Y.append(1 if np.any(labels[i: i + size] == 1) else 0)
    return np.array(X), np.array(Y)


def inspect_dataset(X, y, title="Dataset Check"):
    """可视化函数：帮助你查看数据长什么样"""
    print(f"\n--- {title} ---")
    print(f"数据形状 (样本数, 窗口长度, 特征数): {X.shape}")
    print(f"标签分布 (0为正常, 1为故障): {np.bincount(y.astype(int))}")

    # 随机抽取一个样本的一个特征进行波形可视化
    sample_idx = np.random.randint(0, len(X))
    plt.figure(figsize=(10, 3))
    plt.plot(X[sample_idx, :, 0])  # 绘制第一个传感器特征
    plt.title(f"{title} - Sample {sample_idx} Waveform")
    plt.xlabel("Time steps")
    plt.ylabel("Standardized Value")
    plt.show()


# --- 主逻辑保持原有命名 run_pipeline ---
def run_pipeline():
    # 1. 基础预处理
    # 注意：请确保数据文件夹路径与你的电脑一致
    normal_file = 'MultiphaseFlowFacilitydataset/CVACaseStudy/Training.mat'
    fault_files = [f'MultiphaseFlowFacilitydataset/CVACaseStudy/FaultyCase{i}.mat' for i in range(1, 7)]

    print("正在加载数据...")
    df_normal = load_mat_to_df(normal_file)
    df_normal['label'] = 0

    fault_list = [load_mat_to_df(f) for f in fault_files]
    for df in fault_list:
        df['label'] = 1
    df_all_fault = pd.concat(fault_list, axis=0)

    # 2. 筛选与归一化
    features = df_normal.columns[:-1]
    variances = df_normal[features].var()
    selected_features = variances[variances > 1e-6].index

    scaler = StandardScaler()
    # 严格按照半监督要求：只用正常数据fit
    train_scaled = scaler.fit_transform(df_normal[selected_features])
    fault_scaled = scaler.transform(df_all_fault[selected_features])

    # 3. 滑动窗口
    window_size, step = 60, 20
    print(f"执行滑动窗口 (Size:{window_size}, Step:{step})...")
    X_normal_win, y_normal_win = sliding_window(train_scaled, df_normal['label'].values, window_size, step)
    X_fault_win, y_fault_win = sliding_window(fault_scaled, df_all_fault['label'].values, window_size, step)

    # 4. 【关键修改】修复比例失衡 & 构建真实的测试集
    # 从正常窗口中分出20%作为测试集的“正常对照组”
    X_train, X_test_normal, y_train, y_test_normal = train_test_split(
        X_normal_win, y_normal_win, test_size=0.2, random_state=42
    )

    # 合并测试集：包含正常窗口 + 全部故障窗口
    X_test = np.concatenate([X_test_normal, X_fault_win], axis=0)
    y_test = np.concatenate([y_test_normal, y_fault_win], axis=0)

    # --- 5. 打印检查与结果展示 ---
    print("\n" + "=" * 30)
    print(f"训练集 (纯正常) 形状: {X_train.shape}, 标签分布: {np.bincount(y_train.astype(int))}")
    print(f"测试集 (混合) 形状: {X_test.shape}, 标签分布: {np.bincount(y_test.astype(int))}")
    print("=" * 30)

    # 6. 可视化检查
    inspect_dataset(X_train, y_train, "Training Set (Normal Only)")
    inspect_dataset(X_test, y_test, "Test Set (Mixed)")

    # 7. 保存数据
    save_dir = 'processed_data'
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'train_X.npy'), X_train)
    np.save(os.path.join(save_dir, 'train_y.npy'), y_train)
    np.save(os.path.join(save_dir, 'test_X.npy'), X_test)
    np.save(os.path.join(save_dir, 'test_y.npy'), y_test)
    print(f"\n预处理完成！数据已保存至 {save_dir} 文件夹。")


if __name__ == "__main__":
    run_pipeline()