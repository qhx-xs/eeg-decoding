from scipy import io

import torch
from torch.utils.data  import Dataset
from sklearn.model_selection import train_test_split
import numpy as np


def get_mat_data(file_name:str, cat:str):
    '''
    get data from .mat file.
    ---------
    file_name: the absolute name of .mat file.
    cat: 'xx1_type1' or 'xx1_type2'
    '''
    mat_data = io.loadmat(file_name)
    eeg_data = mat_data[cat]
    print(f'数据路径为：{file_name}, 维度为：{eeg_data.shape}')
    return eeg_data


def choose_action_length(eeg_data, act_start=0, act_end=0, choose_start=0, choose_end=0):
    '''
    对 .mat 文件得到的数据进行选择。
    '''
    channels, data_points, frequency, action_num = eeg_data.shape

    if act_end >= action_num or act_start >= action_num:
        raise ValueError('act_end cannot be bigger than action_num')
    if action_num-act_end <= act_start+1:
        raise ValueError('act尾截太多，仅1个或更少 act！')

    if choose_end >= data_points or choose_start >= data_points:
        raise ValueError('choose_end cannot be bigger than data_points')
    if data_points-choose_end <= choose_start+1:
        raise ValueError('data_points 尾截太多，仅1个或更少 data points！')

    eeg_new = eeg_data[:, choose_start:data_points-choose_end, :, act_start:action_num-act_end]

    return eeg_new


def split_datapoints_4D(eeg_data, step, stride):
    '''
    原始数据是4D的：[channels, datapoints, frequency, number of actions]
    返回 3D 样本列表: [channels, step, frequency]
    '''
    channels, data_points, frequency, action_num = eeg_data.shape

    if step == 'all':
        step = data_points
    if step > data_points:
        raise ValueError(f'划分数据错误，总长{data_points}，子长{step}')

    if not stride > 0:
        raise ValueError('stride should be greater than 0.')

    all_samples = []

    for i in range(action_num):
        sample = eeg_data[:, :, :, i] #[channels, len, freq]
        points_len = sample.shape[1]

        start_step = 0
        end_step = start_step + step

        while end_step < points_len:
            sample_snip = sample[:, start_step:end_step, :]
            sample_snip = torch.tensor(sample_snip, dtype=torch.float)
            all_samples.append(sample_snip)
            start_step += stride
            end_step = start_step + step

        if end_step >= points_len and start_step < points_len: # 保证最后一部分也被包含
            sample_snip = sample[:, points_len-step:, :]
            sample_snip = torch.tensor(sample_snip, dtype=torch.float)
            all_samples.append(sample_snip)

    print(f'样本总数：{len(all_samples)}，维度: {all_samples[0].shape} \n')

    return all_samples


def convert_sample_by_modeltype(samples, modeltype, *args):
    channels, step, frequency = samples[0].shape

    if modeltype in ['cnn', 'cnn_bilstm','cnn_bilstm_attention']: # <--- 修改：新增 'cnn_bilstm'
        print(f'{modeltype} 划分，维度：即为初次划分的维度: [channels, step, frequency]')
        return samples

    elif modeltype == 'lstm':
        samples_lstm = []
        for x in samples:
            # x: [channels, seq_len, frequency]
            x = x.permute(1, 0, 2)  # [seq_len, channels, frequency]
            # 展平最后两个维度
            x = x.reshape(step, -1)  # [seq_len, channels*frequency]
            samples_lstm.append(x)
        print(f'LSTM划分，维度: {samples_lstm[0].shape}')
        return samples_lstm

    elif modeltype == 'lstm-filter':
        if len(args)>=1 and args[0] in [0,1,2,3,4,5]:
            filter_freq = args[0]
            samples_filter = [x[:,:,args[0]] for x in samples]
            samples_filter_new = [x.permute(1, 0) for x in samples_filter]
            print(f'lstm-filter划分，频段 {filter_freq}，维度：{samples_filter_new[0].shape} \n')
            return samples_filter_new
        else:
            raise ValueError('when modeltype is lstm-filter, frequency should be in [0,1,2,3,4,5]')

    else:
        raise ValueError('modeltype is wrong')

def normalize_samples(samples):
    normalized_samples = []
    for sample in samples:
        # 对每个样本的特征维度（时间步）单独标准化
        if len(sample.shape) == 2:  # [seq_len, features] (用于 LSTM)
            mean = sample.mean(dim=0, keepdim=True)
            std = sample.std(dim=0, keepdim=True)
            std = torch.where(std < 1e-8, torch.ones_like(std), std)
            normalized = (sample - mean) / std
        elif len(sample.shape) == 3: # [channels, step, frequency] (用于 CNN)
            # 对整个样本进行标准化
            mean = sample.mean()
            std = sample.std()
            if std < 1e-8:
                std = 1.0
            normalized = (sample - mean) / std
        else:
            normalized = sample
        normalized_samples.append(normalized)
    return normalized_samples

class EEGDataset(Dataset):
    def __init__(self, samples, labels):
        super(EEGDataset, self).__init__()
        self.samples = samples
        self.labels = labels
        if len(self.samples) != len(self.labels):
            raise ValueError('samples is not equal to labels')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

    def split_dataset(self, test_size=0.2, random_state=42):
        # 将 labels 转换为 numpy 数组以进行分层抽样
        labels_np = np.array([l.item() for l in self.labels])

        # 划分数据集，stratify分层抽样
        sam_train, sam_temp, label_train, label_temp = train_test_split(
            self.samples, self.labels, test_size = test_size,
            random_state=random_state, stratify = labels_np)

        # 确保 temp 集合有足够的样本进行二次划分
        if len(sam_temp) < 2:
             raise ValueError("样本总数过少，无法进行 train/eval/test 三分")

        # 将 label_temp 转换为 numpy 数组进行二次分层抽样
        label_temp_np = np.array([l.item() for l in label_temp])

        sam_eval, sam_test, label_eval, label_test = train_test_split(
            sam_temp, label_temp, test_size = 0.5,
            random_state=random_state, stratify = label_temp_np)

        train_dataset = EEGDataset(sam_train, label_train)
        eval_dataset = EEGDataset(sam_eval, label_eval)
        test_dataset = EEGDataset(sam_test, label_test)

        print(f'train num: {len(train_dataset)}')
        print(f'eval num: {len(eval_dataset)}')
        print(f'test num: {len(test_dataset)} \n')

        return train_dataset, eval_dataset, test_dataset