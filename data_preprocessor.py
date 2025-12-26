import scipy.io as sio
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
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


# --- 主逻辑保持 run_pipeline ---
def run_pipeline():
    # 1. 配置路径
    normal_file = 'MultiphaseFlowFacilitydataset/CVACaseStudy/Training.mat'
    fault_files = [f'MultiphaseFlowFacilitydataset/CVACaseStudy/FaultyCase{i}.mat' for i in range(1, 7)]

    # 2.加载与标注
    print("正在加载数据...")
    df_normal = load_mat_to_df(normal_file)
    df_normal['label'] = 0

    fault_list = [load_mat_to_df(f) for f in fault_files]
    for df in fault_list: df['label'] = 1
    df_all_fault = pd.concat(fault_list, axis=0)

    # 3. 筛选与归一化
    features = df_normal.columns[:-1]
    variances = df_normal[features].var()
    selected_features = variances[variances > 1e-6].index

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_normal[selected_features])
    fault_scaled = scaler.transform(df_all_fault[selected_features])

    # 4. 滑动窗口
    window_size, step = 60, 20
    X_normal_win, y_normal_win = sliding_window(train_scaled, df_normal['label'].values, window_size, step)
    X_fault_win, y_fault_win = sliding_window(fault_scaled, df_all_fault['label'].values, window_size, step)

    # 5. 数据集划分 (80%正常用于训练，20%正常+全部故障用于测试)
    X_train, X_test_normal, y_train, y_test_normal = train_test_split(
        X_normal_win, y_normal_win, test_size=0.2, random_state=42
    )
    X_test = np.concatenate([X_test_normal, X_fault_win], axis=0)
    y_test = np.concatenate([y_test_normal, y_fault_win], axis=0)


    # 6. 保存数据文件
    save_dir = 'processed_data'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'train_X.npy'), X_train)
    np.save(os.path.join(save_dir, 'train_y.npy'), y_train)
    np.save(os.path.join(save_dir, 'test_X.npy'), X_test)
    np.save(os.path.join(save_dir, 'test_y.npy'), y_test)
    # 特别注意：需要保存特征名，否则评估脚本不知道热力图的坐标轴叫什么
    np.save(os.path.join(save_dir, 'selected_features.npy'), selected_features.values)

    print(f"数据生产完毕，已存入 {save_dir} 文件夹。")


if __name__ == "__main__":
    run_pipeline()