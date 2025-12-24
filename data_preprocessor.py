import scipy.io as sio
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt  # 新增：用于查看数据
from sklearn.preprocessing import StandardScaler


def load_mat_to_df(file_path):
    mat_data = sio.loadmat(file_path)
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    main_key = max(data_keys, key=lambda k: mat_data[k].size)
    raw_data = mat_data[main_key]
    return pd.DataFrame(raw_data)


def sliding_window(data, labels, size, s):
    X, Y = [], []
    for i in range(0, len(data) - size, s):
        X.append(data[i: i + size])
        Y.append(1 if np.any(labels[i: i + size] == 1) else 0)
    return np.array(X), np.array(Y)


def inspect_dataset(X, y, title="Dataset Check"):
    """
    可视化函数：帮助你查看训练/测试集长什么样
    """
    print(f"\n--- {title} ---")
    print(f"数据形状 (样本数, 窗口长度, 特征数): {X.shape}")
    print(f"标签分布: {np.bincount(y.astype(int))}")

    # 随机抽取一个样本的一个特征进行可视化
    sample_idx = np.random.randint(0, len(X))
    feature_idx = 0
    plt.figure(figsize=(10, 4))
    plt.plot(X[sample_idx, :, feature_idx])
    plt.title(f"{title} - Sample {sample_idx}, Feature {feature_idx}")
    plt.xlabel("Time steps in window")
    plt.ylabel("Normalized Value")
    plt.show()


def save_processed_data(X, y, name, save_dir='processed_data'):
    """
    保存数据为 .npy 格式
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, f'{name}_X.npy'), X)
    np.save(os.path.join(save_dir, f'{name}_y.npy'), y)
    print(f"已保存 {name} 到目录: {save_dir}")


# --- 主逻辑修改 ---
def run_pipeline():
    # 1. 基础预处理（沿用你的逻辑）
    normal_file = 'MultiphaseFlowFacilitydataset/CVACaseStudy/Training.mat'  # 请确保路径正确
    fault_files = [f'MultiphaseFlowFacilitydataset/CVACaseStudy/FaultyCase{i}.mat' for i in range(1, 7)]

    df_normal = load_mat_to_df(normal_file)
    df_normal['label'] = 0

    fault_list = [load_mat_to_df(f) for f in fault_files]
    for i, df in enumerate(fault_list): df['label'] = 1
    df_all_fault = pd.concat(fault_list, axis=0)

    # 2. 筛选与归一化
    features = df_normal.columns[:-1]
    variances = df_normal[features].var()
    selected_features = variances[variances > 1e-6].index

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_normal[selected_features])
    fault_scaled = scaler.transform(df_all_fault[selected_features])

    # 3. 滑动窗口
    X_train, y_train = sliding_window(train_scaled, df_normal['label'].values, 60, 20)
    X_test, y_test = sliding_window(fault_scaled, df_all_fault['label'].values, 60, 20)

    # --- 新增步骤：查看数据 ---
    inspect_dataset(X_train, y_train, "Training Set (Normal)")
    inspect_dataset(X_test, y_test, "Test Set (Faulty)")

    # --- 新增步骤：保存数据 ---
    save_processed_data(X_train, y_train, 'train')
    save_processed_data(X_test, y_test, 'test')

    # 如果你想看具体的数值表（前10行），可以把3D转回2D看一眼
    print("\n训练集第一个样本的前5行数值预览：")
    print(pd.DataFrame(X_train[0][:5], columns=selected_features))


if __name__ == "__main__":
    run_pipeline()