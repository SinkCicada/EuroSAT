# EuroSAT 遥感土地覆盖分类与注意力机制实验

基于经典 ResNet-18，从零训练 baseline、SE、CBAM 三组模型，分析准确率、参数量、卷积/线性层 MACs、模型推理耗时，以及受控亮度和模糊扰动下的表现。应用支持单张 RGB 卫星图块分类和 Grad-CAM 可视化。

本项目使用现成 EuroSAT RGB 数据，无需自行采集或标注。它是图像级分类，不生成地物分割图或检测框。

## 环境

已验证 Python 3.10.22、PyTorch 2.7.1+cu126、torchvision 0.22.1+cu126；其他依赖见 requirements.txt。共享环境已满足依赖时直接使用；安装依赖应在独立虚拟环境执行。

新机器可先根据 GPU 驱动，从 PyTorch 官网安装匹配的 torch/torchvision，再安装其余依赖。requirements.txt 的 CUDA 构建来自实际实验环境，普通 pip 安装是否启用 CUDA 取决于所选索引。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .cache/matplotlib .cache/torch logs results
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TORCH_HOME="$PWD/.cache/torch"
export CUDA_VISIBLE_DEVICES=0
```

运行前执行 `nvidia-smi`，依据当时负载选择 GPU，不能直接假定 GPU 0 空闲。

## 下载与最小验证

```bash
python download.py
python smoke.py
```

下载优先使用 Torchvision 同版本采用的固定提交数据包镜像，并校验 MD5；失败后尝试原始 Hugging Face 地址与作者 Zenodo 地址。只有下载成功且校验通过才将 `.part` 改为正式文件。网络工具需要 `curl`。

- 作者数据记录：https://zenodo.org/records/7711810
- 官方 RGB 包：EuroSAT_RGB.zip，94,658,721 字节，MD5 `f46e308c4d50d4bf32fedad2d3d62f3b`。
- Torchvision 对应包：EuroSAT.zip，94,280,567 字节，MD5 `c8fa014336c82ac7804f0398fcb19387`。
- 实际下载来源和校验信息保存在 `data/download_source.json`。
- 最小验证检查三个模型的前向、有限梯度、一次参数更新、保存和重载一致性。

## 完整实验

```bash
python -u run_experiments.py --epochs 30 --seeds 42 43 44
```

控制器依次执行三个随机种子下的三种模型，全部训练结束后评估测试集。训练集 18,900 张，验证集 4,050 张，测试集 4,050 张；类别分层，划分随机种子固定为 2026。先检查完全重复文件再划分，实际清单及 SHA256 保存在 `data/split.json`。不将数据增强图片单独分入测试集。

训练配置：64×64 RGB 原尺寸输入，保留原始 7×7 stride-2 卷积和最大池化，最终全连接层输出 10 类；AdamW，初始学习率 0.001，权重衰减 0.0001，余弦学习率，batch size 128，交叉熵，CUDA AMP，30 轮。增强为水平/垂直翻转和离散 90° 旋转。三种模型均无预训练权重，不使用测试集选择训练轮次、模型或种子。

可单独训练：

```bash
python train.py --variant baseline --seed 42 --epochs 30
python train.py --variant se --seed 42 --epochs 30
python train.py --variant cbam --seed 42 --epochs 30
python analyze.py
```

`analyze.py` 应在完整实验结束后运行，不用于训练中的调参。它可以从保存的最佳验证集权重重算已有结果。

## 后台运行和进度

```bash
nohup python -u run_experiments.py --epochs 30 --seeds 42 43 44 > logs/experiments.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/experiments.pid
tail -f logs/experiments.log
cat logs/status.json
```

正式部署还记录了启动命令、PID、GPU 快照和环境变量。不要在控制器尚未结束时再次运行完整实验；这会重复占用 GPU，并覆盖同名运行输出。每次独立复现实验可用新项目副本或修改 train.py 的 `--output` 目录。

## 结果与数字依据

- `results/<模型>_seed<种子>/config.json`：参数、环境和训练代码 commit。
- `history.csv`：逐轮训练/验证损失、准确率、学习率和耗时。
- `best.pt`：只按验证集准确率选择的权重；并列保留较早轮次。
- `training_summary.json`：实际总训练耗时。
- `complexity.json`：实际参数量及 Conv2d+Linear MACs。
- `latency.json`：FP32、batch=1，预热30次，测量100次的模型内耗时。
- `results/metrics.json` / `metrics.csv`：逐种子、逐测试条件的真实指标。
- `results/predictions.csv`：每张测试图片的真实类别、预测类别和置信分数。
- `results/figures/`：数据样例、学习曲线、对比图、混淆矩阵和 Grad-CAM。
- `logs/status.json` 中 `state=completed` 才表示训练和评估全部完成。

暗化固定为亮度乘 0.6；模糊固定为 Pillow 高斯模糊 radius=1.0。它们是受控图像扰动，不能等同于真实天气、云层或跨传感器泛化。MACs 只计 Conv2d 和 Linear，不包含池化、归一化、激活、逐元素乘加，不能直接写成完整 FLOPs。共享 GPU 的耗时具有波动，不是独占设备的部署速度保证。

## 单张图块应用

```bash
python predict.py --checkpoint results/cbam_seed42/best.pt \
  --image data/raw/2750/River/River_1.jpg --output results/demo
```

图片路径以实际解压目录为准：可用 `find data/raw -name '*.jpg' | head` 查看。程序输出 10 类预测分数、`prediction.json` 和 `prediction.png`。输入最好为与训练域相近的卫星 RGB 小图块；将任意大图整体缩成64×64不构成大范围制图能力。置信分数未经概率校准，Grad-CAM 是模型解释线索，不是人工地物真值标注。

## 参考资料与边界

1. Helber et al. EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. https://arxiv.org/abs/1709.00029
2. He et al. Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385
3. Hu et al. Squeeze-and-Excitation Networks. https://arxiv.org/abs/1709.01507
4. Woo et al. CBAM: Convolutional Block Attention Module. https://arxiv.org/abs/1807.06521
5. Selvaraju et al. Grad-CAM. https://arxiv.org/abs/1610.02391
6. Torchvision EuroSAT 源码 v0.22.1：https://github.com/pytorch/vision/blob/v0.22.1/torchvision/datasets/eurosat.py

SE/CBAM 是既有方法，本项目的贡献是面向该应用的实现、受控对照和结果分析，不声称提出全新的注意力模块。随机图块划分可能包含地理近邻关联，结果不代表跨地区泛化。三个种子只重复训练随机性，未重复地理划分。
