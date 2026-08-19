# 深度学习服务器环境与 EEG Decoding 运行教程

本文从 Windows 代理开始，完整说明如何在服务器上创建一套独立、可复用于各种神经网络实验的
Conda 环境，然后运行本项目。不要再复用或修改 `isaaclab` 环境。

已验证的服务器信息：

- SSH 别名：`XS-company-server`
- 用户：`haixin`
- GPU：NVIDIA GeForce RTX 5090，32 GB
- 驱动：580.173.02
- `nvidia-smi` 报告的 CUDA：13.0
- Miniforge：`/home/haixin/miniforge3`
- 项目：`/home/haixin/projects/eeg-decoding`
- 新 Conda 环境：`deep-learning`
- Python：3.11
- PyTorch：2.12.0，CUDA 13.0 wheel

## 1. Windows：启动代理

打开 `verge-mihomo`，确认 Mixed/HTTP 端口是 `127.0.0.1:7897`。在 PowerShell 检查：

```powershell
Test-NetConnection 127.0.0.1 -Port 7897
```

应显示 `TcpTestSucceeded : True`。

## 2. Windows：建立服务器反向代理

新开一个 PowerShell，执行并一直保持该窗口运行：

```powershell
ssh -o ExitOnForwardFailure=yes -N -R 7890:127.0.0.1:7897 XS-company-server
```

如果提示服务器的 7890 端口已被占用，通常代表已有代理隧道，不要重复建立。

```text
服务器 127.0.0.1:7890
  -> SSH 反向隧道
  -> Windows 127.0.0.1:7897
  -> verge-mihomo
  -> Internet
```

## 3. Windows：登录服务器并进入 tmux

再开一个 PowerShell：

```powershell
ssh XS-company-server
```

进入可恢复终端，防止本地断网导致安装或训练退出：

```bash
tmux new -As deep-learning
```

## 4. 服务器：设置当前终端代理

每个新服务器终端都需要重新执行：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

检查网络：

```bash
curl -fsSI https://github.com --max-time 15 | head -1
```

## 5. 服务器：clone 或更新项目

第一次运行：

```bash
mkdir -p /home/haixin/projects
cd /home/haixin/projects
git clone https://github.com/qhx-xs/eeg-decoding.git
cd /home/haixin/projects/eeg-decoding
```

如果项目已经 clone 过：

```bash
cd /home/haixin/projects/eeg-decoding
git pull --ff-only
```

## 6. 服务器：创建独立深度学习环境

```bash
source /home/haixin/miniforge3/etc/profile.d/conda.sh
conda env list
cd /home/haixin/projects/eeg-decoding
conda env create -f environment.yml
conda activate deep-learning
```

这会安装 Python 3.11、NumPy、SciPy、scikit-learn、Pandas、Matplotlib、Seaborn、JupyterLab、
TensorBoard 和其他常用工具，但 PyTorch 单独从官方 CUDA wheel 安装。

## 7. 服务器：安装 RTX 5090 对应的 GPU PyTorch

先确认环境正确：

```bash
which python
python -V
conda info --envs
```

Python 路径应包含：

```text
/home/haixin/miniforge3/envs/deep-learning/bin/python
```

安装官方 PyTorch 2.12.0 CUDA 13.0 wheel：

```bash
python -m pip install --upgrade pip
python -m pip install \
  --retries 20 \
  --timeout 1200 \
  torch==2.12.0 \
  torchvision==0.27.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

本项目不需要 torchvision，但保留它方便以后运行图像神经网络。不要再安装系统 CUDA Toolkit；
PyTorch wheel 已携带运行库，服务器驱动负责与 GPU 通信。

检查安装：

```bash
nvidia-smi
python -c 'import torch; print("torch =", torch.__version__); print("torch CUDA =", torch.version.cuda); print("cuda available =", torch.cuda.is_available()); print("GPU =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT AVAILABLE")'
```

预期至少包含：

```text
torch = 2.12.0+cu130
torch CUDA = 13.0
cuda available = True
GPU = NVIDIA GeForce RTX 5090
```

如果 `cuda available = False`，不要继续训练，把完整输出发回来排查。

### 如果下载中途出现 IncompleteRead 或 SSLEOFError

先看命令行最左侧必须是 `(deep-learning)`，不能是 `(base)`。如果仍是 base：

```bash
source /home/haixin/miniforge3/etc/profile.d/conda.sh
conda activate deep-learning
which python
```

然后确认代理变量仍存在：

```bash
env | grep -i proxy
```

`IncompleteRead` 表示几百 MB 的 CUDA 包下载中途断线，不是版本错误。保持 Windows 的 SSH 反向代理窗口
运行，重新执行上面带 `--retries 20 --timeout 1200` 的安装命令即可。安装结束前不要切回 base 环境。

## 8. 服务器：从原始 EDF 重新生成 1 秒数据

原始 EDF 已上传到服务器时，先确认目录并创建数据目录：

```bash
mkdir -p /home/haixin/projects/eeg-decoding/data
find /你的原始数据目录 -maxdepth 1 -iname '*.edf' | head
```

每个 4 秒刺激/想象阶段会切成四个互不重叠的 1 秒窗：

```bash
python preprocess_edf.py \
  --input /你的原始数据目录 \
  --output data/sub03_eeg_1s.pt \
  --config configs/preprocess.json
```

预期是 640 个窗口，形状 `[640,8,100,6]`：

```bash
ls -lh data/sub03_eeg_1s.pt
```

## 9. 以后每次登录后的初始化

环境只创建一次。每个新终端只执行：

```bash
source /home/haixin/miniforge3/etc/profile.d/conda.sh
conda activate deep-learning
cd /home/haixin/projects/eeg-decoding
```

安全回读 PT：

```bash
python -c 'from pathlib import Path; from eeg_pipeline.training import load_pt_dataset; d=load_pt_dataset(Path("data/sub03_eeg_1s.pt")); print(d["features"].shape); print("finite =", d["features"].isfinite().all().item())'
```

预期：

```text
torch.Size([640, 8, 100, 6])
finite = True
```

## 10. 服务器：运行测试

```bash
python -m unittest discover -s tests -v
```

应显示所有测试和 `OK`。

## 11. 服务器：全局搜索并重复正式训练

四分类使用固定随机窗口 70%/15%/15% 切分，搜索 60 组参数，再用最佳参数训练 400 轮、重复 5 次：

```bash
python global_search.py --data data/sub03_eeg_1s.pt --task four_class \
  --trials 60 --search-epochs 100 --final-epochs 400 --repeats 5
```

想象二分类：

```bash
python global_search.py --data data/sub03_eeg_1s.pt --task imagery_binary \
  --trials 60 --search-epochs 100 --final-epochs 400 --repeats 5
```

搜索可在 tmux 中运行；同一命令中断后可继续。查看最终结果：

```bash
cat outputs/global_search/four_class/search_summary.json
cat outputs/global_search/imagery_binary/search_summary.json
```

结果目录：

```text
outputs/global_search/four_class/
outputs/global_search/imagery_binary/
```

## 12. tmux 常用操作

训练中暂时退出但保持程序运行：按 `Ctrl+B`，松开，再按 `D`。

重新连接：

```bash
ssh XS-company-server
tmux attach -t deep-learning
```

查看 GPU：

```bash
watch -n 2 nvidia-smi
```

## 最短记忆版

```text
Windows：verge-mihomo -> 保持 ssh -R 反向代理
服务器：source conda.sh -> conda activate deep-learning
项目：cd /home/haixin/projects/eeg-decoding -> git pull
数据：服务器原始 EDF -> preprocess_edf.py -> sub03_eeg_1s.pt
检查：torch.cuda.is_available() 必须为 True，测试必须 OK
训练：python global_search.py --data data/sub03_eeg_1s.pt --task four_class --trials 60 --search-epochs 100 --final-epochs 400 --repeats 5
```

PyTorch 安装命令依据官方版本页：
https://pytorch.org/get-started/previous-versions/
