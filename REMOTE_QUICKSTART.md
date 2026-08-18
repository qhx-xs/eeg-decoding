# EEG Decoding 服务器最简运行教程

本文针对当前已经验证过的服务器账号和 IsaacLab Conda 环境，说明如何把项目代码与本机生成的
PT 数据放到服务器，然后运行测试和 GPU 训练。

当前环境信息：

- SSH 别名：`XS-company-server`
- 服务器用户：`haixin`
- 项目目录：`/home/haixin/projects/eeg-decoding`
- Conda 初始化脚本：`/home/haixin/miniforge3/etc/profile.d/conda.sh`
- Conda 环境：`isaaclab`
- IsaacLab 目录：`/home/haixin/projects/IsaacLab`

这个 EEG 项目只使用 IsaacLab 环境中的 Python、PyTorch 和 CUDA，不启动 Isaac Sim，因而不需要
设置 `ISAACLAB_ROOT`、`OMNI_KIT_ACCEPT_EULA`、资产缓存或 MIND 的 `PYTHONPATH`，也不需要通过
`isaaclab.sh -p` 启动。

## 1. Windows：需要联网时建立服务器代理

如果服务器可以直接访问 GitHub，可以跳过本节。否则先启动 Windows 上的 `verge-mihomo`，确认
Mixed/HTTP 端口为 `127.0.0.1:7897`：

```powershell
Test-NetConnection 127.0.0.1 -Port 7897
```

再新开一个 PowerShell，并保持该窗口运行：

```powershell
ssh -o ExitOnForwardFailure=yes -N -R 7890:127.0.0.1:7897 XS-company-server
```

## 2. Windows：登录服务器

```powershell
ssh XS-company-server
```

建议进入 tmux，防止本地断网导致训练退出：

```bash
tmux new -As eeg-decoding
```

## 3. 服务器：clone 项目

如使用上一节的反向代理，先在当前服务器终端执行：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

然后 clone：

```bash
mkdir -p /home/haixin/projects
cd /home/haixin/projects
git clone https://github.com/qhx-xs/eeg-decoding.git
cd /home/haixin/projects/eeg-decoding
```

如果目录已经存在，不要重复 clone，改用：

```bash
cd /home/haixin/projects/eeg-decoding
git pull --ff-only
```

## 4. 服务器：激活 IsaacLab Python 环境

每次新开服务器终端都执行：

```bash
source /home/haixin/miniforge3/etc/profile.d/conda.sh
conda activate isaaclab
cd /home/haixin/projects/eeg-decoding
```

检查 Python、PyTorch 和 GPU：

```bash
which python
python -V
nvidia-smi
python -c 'import torch; print("torch =", torch.__version__); print("cuda =", torch.cuda.is_available()); print("gpu =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT AVAILABLE")'
```

训练前必须看到 `cuda = True`。

### 不要直接安装普通 requirements.txt

IsaacLab 对 PyTorch、CUDA 和 NumPy 版本有自己的约束。不要在 `isaaclab` 环境里运行
`pip install -r requirements.txt`，否则可能升级这些核心包并破坏 IsaacLab。

先检查 EEG 项目需要的包：

```bash
python -c 'import torch, numpy, scipy, sklearn; print("DEPENDENCIES_OK")'
```

如果提示缺少 SciPy 或 scikit-learn，只安装服务器补充依赖，不会主动安装或升级 Torch/NumPy：

```bash
python -m pip install -r requirements-isaaclab.txt
```

再次执行上面的 `DEPENDENCIES_OK` 检查。

## 5. Windows：上传本机生成的 PT

先在服务器创建数据目录：

```bash
mkdir -p /home/haixin/projects/eeg-decoding/data
```

然后在 Windows 上另开 PowerShell（不是服务器终端）执行：

```powershell
scp "C:\Users\QHX\Desktop\cnn_lstm\data\sub03_eeg.pt" "XS-company-server:/home/haixin/projects/eeg-decoding/data/sub03_eeg.pt"
```

回到服务器检查文件，大小应约为 18 MB：

```bash
ls -lh /home/haixin/projects/eeg-decoding/data/sub03_eeg.pt
```

安全回读并检查形状：

```bash
cd /home/haixin/projects/eeg-decoding
python -c 'from pathlib import Path; from eeg_pipeline.training import load_pt_dataset; d=load_pt_dataset(Path("data/sub03_eeg.pt")); print(d["features"].shape); print("finite =", d["features"].isfinite().all().item())'
```

预期输出：

```text
torch.Size([480, 8, 200, 6])
finite = True
```

## 6. 服务器：运行轻量测试

```bash
cd /home/haixin/projects/eeg-decoding
python -m unittest discover -s tests -v
```

应显示 `Ran 8 tests` 和 `OK`。测试只使用合成数据，不启动完整训练。

## 7. 服务器：运行正式训练

建议一次只运行一个实验。正式的跨 run 四分类：

```bash
cd /home/haixin/projects/eeg-decoding
python train.py --data data/sub03_eeg.pt --task four_class --cv run
```

正式的跨 run 想象二分类：

```bash
python train.py --data data/sub03_eeg.pt --task imagery_binary --cv run
```

辅助的 trial 五折实验：

```bash
python train.py --data data/sub03_eeg.pt --task four_class --cv trial
python train.py --data data/sub03_eeg.pt --task imagery_binary --cv trial
```

结果分别保存在：

```text
outputs/four_class_run/
outputs/imagery_binary_run/
outputs/four_class_trial/
outputs/imagery_binary_trial/
```

每个目录包含各折最佳模型和汇总指标 `metrics.json`。

## 8. tmux 最常用操作

训练过程中暂时离开但保持任务运行：按 `Ctrl+B`，松开，再按 `D`。

重新连接：

```bash
ssh XS-company-server
tmux attach -t eeg-decoding
```

查看显卡使用情况：

```bash
watch -n 2 nvidia-smi
```

## 最短记忆版

```text
服务器：cd /home/haixin/projects/eeg-decoding -> git pull
环境：source conda.sh -> conda activate isaaclab
数据：Windows 用 scp 上传 sub03_eeg.pt 到项目 data/
检查：测试通过且 torch.cuda.is_available() 为 True
训练：python train.py --data data/sub03_eeg.pt --task four_class --cv run
```
