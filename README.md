# 多相流设施数据集异常检测项目

## 1. 任务背景

本项目是天津大学25261数据挖掘小组的实践项目，旨在对CMFF（Cranfield Multiphase Flow Facility）多相流设施数据集进行故障异常分类检测。通过机器学习方法实现对工业过程监控数据的异常检测，为工业设备故障预警提供技术支持。

## 2. 数据集介绍

### 数据集来源

- **数据集名称**: Cranfield Multiphase Flow Facility (CMFF) Dataset
- **数据格式**: MATLAB .mat 文件
- **数据内容**: 包含1个正常工况数据和6个故障工况数据
- **数据特征**: 多传感器时间序列数据

### 文件结构

```
MultiphaseFlowFacilitydataset/
├── CVACaseStudy/
│   ├── Training.mat          # 正常工况数据
│   ├── FaultyCase1-6.mat     # 6个故障工况数据
│   └── license.txt           # 数据集许可证
└── Statistical process monitoring of a multiphase flow facility.pdf  # 数据集说明文档
```

### 数据预处理

- 使用滑动窗口技术处理时间序列数据（窗口大小60，步长20）
- 特征选择：基于方差筛选有效特征
- 数据标准化：使用StandardScaler进行归一化
- 数据集划分：80%正常数据用于训练，20%正常数据+全部故障数据用于测试

## 3. 依赖安装

### 环境要求

- Python 3.12.7

### 安装依赖

```bash
uv sync .
```

### 主要依赖包

- numpy>=2.4.0
- pandas>=2.3.3
- scikit-learn>=1.8.0
- scipy>=1.16.3
- tensorflow>=2.20.0
- torch>=2.9.1
- matplotlib>=3.10.8
- seaborn>=0.13.2
- mlxtend>=0.24.0

## 4. 使用说明

### 数据预处理

首先运行数据预处理脚本生成训练和测试数据：

```bash
python data_preprocessor.py
```

### 运行异常检测算法

#### LSTM自编码器方法

```bash
python algorithm_model/LSTM-Autoencoder.py
```

#### Isolation Forest方法

- 基准版本：

```bash
python algorithm_model/unsupervised_iforest.py
```

- 优化版本：

```bash
python algorithm_model/unsupervised_iforest_optimized.py
```

### 数据评估

运行数据评估脚本查看数据特征分析：

```bash
python data_evaluator.py
```

## 5. 相关脚本说明

### 核心脚本

- **data_preprocessor.py**: 数据预处理脚本，负责数据加载、特征选择、标准化和数据集划分
- **data_evaluator.py**: 数据评估脚本，提供数据统计分析和可视化

### 算法模型

- **LSTM-Autoencoder.py**: LSTM自编码器异常检测算法（无监督学习）
- **lstm_ae_unsupervised.py**: LSTM自编码器无监督版本
- **unsupervised_iforest.py**: Isolation Forest基准算法
- **unsupervised_iforest_optimized.py**: Isolation Forest优化版本
- **Metric Classifier.py**: 度量分类器算法

### 结果文件

- **processed_data/**: 预处理后的数据文件
- **evaluation_results/**: 算法评估报告
- **image_results/**: 结果可视化图片

## 6. 结果说明

### LSTM自编码器结果

- **准确率**: 98.82%
- **精确率**: 99.71%
- **召回率**: 99.05%
- **阈值**: 0.841437 (95%置信度)

### Isolation Forest结果

- 提供基准版本和优化版本对比
- 包含异常分数分布可视化
- 详细的分类性能报告

### 可视化结果

项目生成多种可视化图表：

- 训练集和测试集数据分布
- 传感器相关性热力图
- 算法训练损失曲线
- 异常检测结果分布图

## 7. 引用许可

### 项目许可证

本项目代码使用 **MIT许可证**：

```
Copyright (c) 2025 Liu Linyu
```

### 数据集许可证

数据集使用类似BSD的许可证：

```
Copyright (c) 2015, Yi Cao
```

### 引用要求

使用本项目或数据集时，请遵守相应的许可证要求，并适当引用原始数据来源。

## 项目结构

```
.
├── algorithm_model/          # 算法模型实现
├── evaluation_results/       # 评估结果报告
├── image_results/           # 可视化结果图片
├── processed_data/          # 预处理后的数据
├── MultiphaseFlowFacilitydataset/  # 原始数据集
├── data_preprocessor.py     # 数据预处理脚本
├── data_evaluator.py        # 数据评估脚本
├── pyproject.toml          # 项目配置和依赖
└── LICENSE                 # 项目许可证
```

## 联系方式

如有问题或建议，请联系项目维护者。
