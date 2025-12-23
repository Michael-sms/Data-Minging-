import scipy.io as sio
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


def load_mat_to_df(file_path):
    """
    自适应读取mat文件，自动识别数据矩阵键名
    """
    mat_data = sio.loadmat(file_path)
    # 过滤掉mat文件自带的元数据键（以__开头）
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]

    # 通常CMFF数据存储在最大的矩阵中
    main_key = max(data_keys, key=lambda k: mat_data[k].size)
    raw_data = mat_data[main_key]

    # 转换为DataFrame
    df = pd.DataFrame(raw_data)
    # 假设CMFF数据集典型的24-26个过程变量
    return df


def preprocess_cmff_dataset(normal_path, fault_paths, window_size=100, step=50):
    """
    完整预处理流程：加载 -> 筛选 -> 归一化 -> 标注 -> 窗口化
    """
    # 1. 加载数据
    print("正在加载数据...")
    df_normal = load_mat_to_df(normal_path)
    df_normal['label'] = 0

    fault_list = []
    for f_path in fault_paths:
        df_f = load_mat_to_df(f_path)
        # 这里进行单标签标注：所有故障文件数据均设为 1
        df_f['label'] = 1
        fault_list.append(df_f)

    df_all_fault = pd.concat(fault_list, axis=0)

    # 2. 信号筛选 (Feature Selection)
    # 剔除方差极小（接近0）的传感器，这些通常是死掉的传感器或固定设定值
    features = df_normal.columns[:-1]
    variances = df_normal[features].var()
    selected_features = variances[variances > 1e-6].index
    print(f"原变量数: {len(features)}, 筛选后保留变量数: {len(selected_features)}")

    # 3. 归一化 (Standardization)
    # 仅使用正常数据拟合，防止测试数据信息泄露
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(df_normal[selected_features])
    fault_data_scaled = scaler.transform(df_all_fault[selected_features])

    # 4. 滑动窗口切片 (Sliding Window)
    def sliding_window(data, labels, size, s):
        X, Y = [], []
        for i in range(0, len(data) - size, s):
            X.append(data[i: i + size])
            # 窗口内只要有1个点是故障，则该窗口标为1
            Y.append(1 if np.any(labels[i: i + size] == 1) else 0)
        return np.array(X), np.array(Y)

    print("正在进行滑动窗口切片...")
    X_train, y_train = sliding_window(train_data_scaled, df_normal['label'].values, window_size, step)
    X_test_fault, y_test_fault = sliding_window(fault_data_scaled, df_all_fault['label'].values, window_size, step)

    return X_train, y_train, X_test_fault, y_test_fault, selected_features


# --- 执行预处理 ---
# 确保文件名与您上传的一致
normal_file = 'MultiphaseFlowFacilitydataset/CVACaseStudy/Training.mat'
fault_files = [f'MultiphaseFlowFacilitydataset/CVACaseStudy/FaultyCase{i}.mat' for i in range(1, 7)]

X_train, y_train, X_test_f, y_test_f, feat_names = preprocess_cmff_dataset(
    normal_file, fault_files, window_size=60, step=20
)

print(f"最终训练集形状 (Samples, Window, Features): {X_train.shape}")
print(f"最终故障测试集形状: {X_test_f.shape}")

# 保存预处理后的数据(保存到preprocessed_data文件夹下)
if not os.path.exists('preprocessed_data'):
    os.makedirs('preprocessed_data')
np.savez_compressed('preprocessed_data/cmff_data.npz', X_train=X_train, y_train=y_train, X_test_f=X_test_f, y_test_f=y_test_f, feat_names=feat_names)
