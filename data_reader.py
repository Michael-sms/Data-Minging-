import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# 查找所有 .mat 文件
dataset_path = 'MultiphaseFlowFacilitydataset/CVACaseStudy'
mat_files = [f for f in os.listdir(dataset_path) if f.endswith('.mat')]

# 读取并分析数据
for mat_file in mat_files:
    file_path = os.path.join(dataset_path, mat_file)
    print(f"\n处理文件: {mat_file}")

    try:
        # 读取文件
        data = sio.loadmat(file_path, squeeze_me=True)

        # 打印文件信息
        print(f"变量列表: {list(data.keys())}")

    except Exception as e:
        print(f"读取 {mat_file} 时出错: {str(e)}")


# 批量处理
def batch_process_mat_files(folder_path):
    """批量处理文件夹中的所有 .mat 文件"""
    all_data = {}

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mat'):
                full_path = os.path.join(root, file)
                try:
                    data = sio.loadmat(full_path, squeeze_me=True)
                    all_data[file] = {
                        'path': full_path,
                        'variables': list(data.keys()),
                        'data': data
                    }
                    print(f"成功读取: {file}")
                except Exception as e:
                    print(f"读取失败 {file}: {e}")

    return all_data


# 运行批量处理
dataset = batch_process_mat_files('MultiphaseFlowFacilitydataset')
print(f"总共读取了 {len(dataset)} 个 .mat 文件")