# -*- coding: utf-8 -*-
"""
主程序：使用 EEG_CNN_BiLSTM 模型进行四分类任务
"""
import os
from pathlib import Path
import torch
import torch.optim as optim

# 导入数据处理函数
from get_dataset import get_mat_data
from get_dataset import choose_action_length
from get_dataset import split_datapoints_4D
from get_dataset import convert_sample_by_modeltype
from get_dataset import EEGDataset, normalize_samples

# 导入模型和损失函数
from EEG_Model import EEG_CNN_BiLSTM, FocalLoss,EEG_CNN_BiLSTM_Attention

# 导入训练函数
from funs_for_epoch import classify_eeg_4class


#%% get dataset and set parameters.

# ==============================================================================
# 路径和数据参数 (请根据您的实际环境修改文件路径)
# ==============================================================================
legacy_data_dir = Path(__file__).resolve().parent / 'legacy_data'
filename1 = str(legacy_data_dir / 'jingQing1.mat')
filename2 = str(legacy_data_dir / 'jingQing2.mat')
filename3 = str(legacy_data_dir / 'jingZhong1.mat')
filename4 = str(legacy_data_dir / 'jingZhong2.mat')

person = 'JZW'
act_start = 20    # 从第20个动作开始取
act_end = 0       # 动作结尾不截取
choose_start = 50 # 单个动作内前50个数据点不取
choose_end = 0    # 单个动作内结尾不截取
step_all = [400]  # 样本时间步长
stride = 100      # 滑动步长

modeltype = 'cnn_bilstm_attention' # <--- 使用 EEG_CNN_BiLSTM
freq = [0]               # 频率参数 (在 CNN-BiLSTM 中通常不使用，但保留参数结构)

# ==============================================================================
# 训练参数
# ==============================================================================
learning_rate = 0.0006
l2_rate = 0.003
num_epoch = 800
batch = [32, 32, 32] # 训练、验证、测试的批次大小


for step in step_all:

    uniform_name = f'{person}_{act_start}_{act_end}_{choose_start}_{choose_end}_{step}_{stride}_{modeltype}_{num_epoch}_4class'

    # --- 数据加载、预处理与标签分配 ---

    # 类别 0: jingQing1
    eeg1 = get_mat_data(filename1, 'xx1_type1')
    eeg1 = choose_action_length(eeg1, act_start, act_end, choose_start, choose_end)
    samples1 = split_datapoints_4D(eeg1, step, stride)
    # 使用 'cnn_bilstm' 转换，保持 3D 形状 [channels, step, frequency]
    samples1 = convert_sample_by_modeltype(samples1, modeltype, freq)
    samples1 = normalize_samples(samples1)
    labels1 = [torch.tensor(0, dtype=torch.long) for _ in range(len(samples1))]

    # 类别 1: jingQing2
    eeg2 = get_mat_data(filename2, 'xx1_type2')
    eeg2 = choose_action_length(eeg2, act_start, act_end, choose_start, choose_end)
    samples2 = split_datapoints_4D(eeg2, step, stride)
    samples2 = convert_sample_by_modeltype(samples2, modeltype, freq)
    samples2 = normalize_samples(samples2)
    labels2 = [torch.tensor(1, dtype=torch.long) for _ in range(len(samples2))]

    # 类别 2: jingZhong1
    eeg3 = get_mat_data(filename3, 'xx1_type1')
    eeg3 = choose_action_length(eeg3, act_start, act_end, choose_start, choose_end)
    samples3 = split_datapoints_4D(eeg3, step, stride)
    samples3 = convert_sample_by_modeltype(samples3, modeltype, freq)
    samples3 = normalize_samples(samples3)
    labels3 = [torch.tensor(2, dtype=torch.long) for _ in range(len(samples3))]

    # 类别 3: jingZhong2
    eeg4 = get_mat_data(filename4, 'xx1_type2')
    eeg4 = choose_action_length(eeg4, act_start, act_end, choose_start, choose_end)
    samples4 = split_datapoints_4D(eeg4, step, stride)
    samples4 = convert_sample_by_modeltype(samples4, modeltype, freq)
    samples4 = normalize_samples(samples4)
    labels4 = [torch.tensor(3, dtype=torch.long) for _ in range(len(samples4))]

    # --- 合并所有数据和划分数据集 ---
    all_samples = samples1 + samples2 + samples3 + samples4
    all_labels = labels1 + labels2 + labels3 + labels4

    dataset = EEGDataset(all_samples, all_labels)
    train_dataset, eval_dataset, test_dataset = dataset.split_dataset()

    print(f"\n=== 四分类数据统计 ===")
    print(f"jingQing1 (类别0): {len(samples1)} 样本")
    print(f"jingQing2 (类别1): {len(samples2)} 样本")
    print(f"jingZhong1 (类别2): {len(samples3)} 样本")
    print(f"jingZhong2 (类别3): {len(samples4)} 样本")
    print(f"总样本数: {len(all_samples)}")
    print(f"训练集: {len(train_dataset)}, 验证集: {len(eval_dataset)}, 测试集: {len(test_dataset)}")


    #%% run model.
    # 样本维度: [channels, step, frequency]
    channels, sam_len, input_dim = all_samples[0].shape

    # 实例化 EEG_CNN_BiLSTM 模型
    model = EEG_CNN_BiLSTM_Attention(channels=channels, num_classes=4)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # 使用 Focal Loss，根据混淆矩阵调整权重
    # 将 class_weights 移动到与模型相同的设备
    class_weights = torch.tensor([1.3, 1.1, 1.5, 1.5]).to(device)
    loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)

    # 优化器和调度器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) # 未在训练函数中使用

    # 使用四分类训练函数开始训练
    classify_eeg_4class(uniform_name, train_dataset, eval_dataset, test_dataset, batch, model, loss_fn, optimizer, num_epoch)
