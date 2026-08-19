# EDF 六频带 CNN-GRU-Transformer-Attention

本项目把带 EDF+ trigger 的 EEG 记录转换成六频带时序特征，并使用
CNN-GRU-Transformer-Attention 完成四分类与想象二分类。CNN 提取局部时频特征，双向 GRU
建模短程顺序，Transformer Encoder 建模全局依赖，最后使用可学习注意力池化分类。
数据、模型权重和带本机路径的报告不会提交到 Git。

## 数据定义

- EEG 通道：`O1, Oz, O2, Pz, C3, C4, Cz, Fz`
- trigger：41/42 为苹果/锤子刺激，51/52 为苹果/锤子视觉想象
- 特征：delta、theta、alpha、low-beta、high-beta、low-gamma 的 log-Hilbert power
- 每个 4 秒阶段切成四个互不重叠的 1 秒窗，形成 `[8, 100, 6]` 输入
- `.pt` 同时保存四分类标签、物体标签、phase、run/trial 分组及 QC 信息

旧的 `jingQing1.mat` 第六个特征出现约 `1e12` 的异常值，本管线不复制该预处理结果。

## 服务器环境

已验证服务器请直接按照 [REMOTE_QUICKSTART.md](REMOTE_QUICKSTART.md) 操作。该文档会创建独立的
`deep-learning` Conda 环境，不会修改 IsaacLab，并包含代理、clone、RTX 5090 GPU PyTorch、PT 上传和训练步骤。

建议 Python 3.11 或 3.12，并根据服务器 CUDA 版本先从
[PyTorch 官方安装页](https://pytorch.org/get-started/locally/)安装对应的 PyTorch，再安装其余依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 预处理

```bash
python preprocess_edf.py \
  --input /path/to/Sub03_XHQRaw \
  --output data/sub03_eeg_1s.pt \
  --config configs/preprocess.json
```

期望输出为 4 个 run、80 个 trial、160 个阶段、640 个窗口，特征形状
`[640, 8, 100, 6]`。脚本同时生成不含本机绝对路径的 summary JSON。

## 训练

纯随机窗口切分使用 70%/15%/15% 的训练、验证、测试比例：

```bash
python train.py --data data/sub03_eeg_1s.pt --task four_class --cv random
python train.py --data data/sub03_eeg_1s.pt --task imagery_binary --cv random
```

注意：不重叠窗口没有重复采样点，但同一 trial 的窗口仍可能被分到不同集合，因此随机窗口结果可能
偏乐观，适合探索模型，不应表述为跨 trial 或跨 run 泛化性能。原有分组切分仍可用于对照：

```bash
python train.py --data data/sub03_eeg_1s.pt --task four_class --cv run
python train.py --data data/sub03_eeg_1s.pt --task imagery_binary --cv run
```

辅助的按 trial 五折结果：

```bash
python train.py --data data/sub03_eeg_1s.pt --task four_class --cv trial
python train.py --data data/sub03_eeg_1s.pt --task imagery_binary --cv trial
```

默认固定训练 400 轮，不早停；训练结束后仍选择验证集 phase/trial macro-F1 最高的那一轮作为
该折模型，测试集不参与选模型。可临时指定轮数和实验名：

```bash
python train.py --data data/sub03_eeg_1s.pt --task four_class --cv random \
  --epochs 300 --run-name four_class_300
```

主要指标先平均同一阶段落入当前集合的窗口 logits，窗口级指标仅作为辅助。每次训练会在独立输出目录保存：

- `metrics.json`：逐折指标、均值、标准差和完整训练历史
- `learning_curves.png`：训练损失和验证集 Macro-F1 曲线
- `confusion_phase_or_trial.png`：主要结果的聚合混淆矩阵（计数及按行归一化）
- `confusion_window.png`：辅助的窗口级混淆矩阵
- `fold_*_confusion_*.png`：每一折的混淆矩阵
- 对应的 `confusion_*.csv` 及每折最优 checkpoint

训练配置位于 `configs/train.json`，默认要求 CUDA；若只做合成调试，可复制配置并把 `device`
改成 `cpu`。

## 全局 Optuna 搜索与重复训练

以下命令固定一次随机 70/15/15 切分；测试集在搜索期间完全不可见。TPE 同时搜索优化器、CNN、
GRU、Transformer、dropout 与 label smoothing，随后使用最佳配置训练 400 轮并以 5 个随机种子
重复训练，最终报告均值、标准差、混淆矩阵和每次 checkpoint：

```bash
python global_search.py --data data/sub03_eeg_1s.pt --task four_class \
  --trials 60 --search-epochs 100 --final-epochs 400 --repeats 5

python global_search.py --data data/sub03_eeg_1s.pt --task imagery_binary \
  --trials 60 --search-epochs 100 --final-epochs 400 --repeats 5
```

搜索可中断后用同一命令续跑，结果位于 `outputs/global_search/<task>/`。`search_summary.json`
保存最佳参数和最终测试汇总，`final/metrics.json` 保存每次训练的完整历史。

## 分组嵌套搜索（保留作严谨对照）

`search.py` 用嵌套 TPE 贝叶斯优化搜索学习率、权重衰减、batch size、CNN/GRU/Transformer 结构和 dropout
和 label smoothing。每个外层测试 fold 都单独在其训练/验证数据上搜索参数，当前测试 fold 在搜索
期间完全不可见。这样不会因为同一个 run 在另一折充当验证集而间接泄漏测试信息。搜索结束后会
用每个外层 fold 各自的最佳参数固定训练 200 轮并生成上述混淆矩阵：

```bash
python search.py --data data/sub03_eeg_1s.pt --task four_class --cv run \
  --trials 24 --search-epochs 60 --final-epochs 200

python search.py --data data/sub03_eeg_1s.pt --task imagery_binary --cv run \
  --trials 24 --search-epochs 60 --final-epochs 200
```

搜索状态保存在 `outputs/search_nested/<task>_<cv>/study.db`，中断后运行同一条命令会继续，
而不是从头搜索。`best_configs.json`、`fold_*_trials.json` 和 `search_summary.json` 保存各外层
fold 的最佳参数、全部候选与分数。

## 测试

测试不读取真实 EDF，也不会启动完整训练：

```bash
python -m unittest discover -s tests -v
```

`classifier_main_3.py`、`get_dataset.py` 和 `funs_for_epoch.py` 是旧 MAT 工作流；新实验使用
`preprocess_edf.py` 与 `train.py`。
