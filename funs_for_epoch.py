import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
# from tqdm import tqdm  # <--- 已移除 tqdm 以去掉进度条

# 保存模型文件的路径，使用 os.path.join 保证路径兼容性
# 注意：主程序中通过参数传入 save_path 更健壮，这里使用硬编码作为默认值，但建议主程序传入
SAVE_PATH = str(Path(__file__).resolve().parent / 'outputs' / 'legacy')


def classify_eeg_4class(uniform_name, train_dataset, eval_dataset, test_dataset, batch, model, loss_fn, optimizer, num_epoch, save_path=SAVE_PATH): # <--- 接受 save_path 参数
    '''
    四分类模型的训练、验证和测试。
    '''

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True) # <--- 确保文件夹存在

    # 确保模型、损失函数在正确的设备上
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn.to(device)

    # 设置 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch[0], shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch[1], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch[2], shuffle=False)

    print(f"当前设备: {device}")

    best_eval_acc = 0.0

    for epoch in range(num_epoch):
        model.train()
        train_loss_total = 0.0

        # === 训练阶段 (已移除 tqdm 进度条) ===
        for samples, labels in train_loader:
            samples, labels = samples.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(samples)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        avg_train_loss = train_loss_total / len(train_loader)

        # === 验证阶段 ===
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for samples, labels in eval_loader:
                samples, labels = samples.to(device), labels.to(device)
                outputs = model(samples)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        eval_acc = correct / total

        # 打印并保存最佳模型
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_model_name = os.path.join(save_path, f'best_model_{uniform_name}.pt') # 使用 os.path.join
            torch.save(model.state_dict(), best_model_name)
            print(f'*** 最佳模型更新：Acc={best_eval_acc:.4f}, 已保存至 {best_model_name} ***')

        print(f'Epoch [{epoch+1}/{num_epoch}], Train Loss: {avg_train_loss:.4f}, Eval Acc: {eval_acc:.4f}')


    # === 测试阶段 ===
    print("\n--- 开始最终测试 ---")

    # 加载最佳模型
    load_model_path = os.path.join(save_path, f'best_model_{uniform_name}.pt')
    if not os.path.exists(load_model_path):
        print(f"错误：未找到最佳模型文件 {load_model_path}。请检查训练过程是否成功保存模型。")
        return 0.0

    model.load_state_dict(torch.load(load_model_path))
    model.eval()

    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        # 测试阶段也移除 tqdm 进度条
        for samples, labels in test_loader:
            samples, labels = samples.to(device), labels.to(device)
            outputs = model(samples)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = correct / total

    # 计算原始混淆矩阵 (计数)
    cm_count = confusion_matrix(all_labels, all_predicted)

    # 将混淆矩阵归一化为正确率 (沿行归一化，即除以真实标签的总数)
    cm_accuracy = cm_count.astype('float') / cm_count.sum(axis=1)[:, np.newaxis]

    # 将 NaN 替换为 0 (防止某个类别的真实样本数为 0 时出现 NaN)
    cm_accuracy = np.nan_to_num(cm_accuracy)

    print(f'测试集总准确率: {test_acc:.4f}')
    print('\n归一化混淆矩阵 (按真实标签的正确率):')
    print(np.round(cm_accuracy, 4)) # 打印时保留 4 位小数

    # 保存结果
    result_filename = os.path.join(save_path, f'result_{uniform_name}.txt')
    with open(result_filename, 'w') as f:
        f.write(f'最终测试准确率: {test_acc:.4f}\n')
        f.write('归一化混淆矩阵 (按真实标签的正确率):\n')
        f.write(np.array2string(cm_accuracy, precision=4) + '\n') # 保存时也保留 4 位小数
    print(f'结果已保存到 {result_filename}')

    return test_acc
