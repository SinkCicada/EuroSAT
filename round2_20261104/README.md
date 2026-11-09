# Round 2 · EuroSAT 小图输入适配与注意力复验

服务器训练源码提交 `57db63b50240e1b5fc04ba14c8c436fded6a8095`。本目录的 Python 文件为实际训练源码快照。环境见仓库根目录，第二轮完整控制器需要 CUDA。在新克隆副本中从根目录运行，检查 `nvidia-smi` 后按实时占用设置 `CUDA_VISIBLE_DEVICES`。

```bash
python download.py
cd round2_20261104
python train.py --variant small --seed 42 --epochs 1 --data ../data --output smoke_check --smoke
mkdir -p logs
nohup python -u controller.py > logs/run.log 2>&1 < /dev/null &
echo $! > logs/run.pid
```

等待 `logs/status.json` 的 `state` 为 `completed`，再补跑原版对照。请勿在旧的运行目录重复启动控制器。

```bash
nohup python -u extend_baseline.py > logs/extension.log 2>&1 < /dev/null &
echo $! > logs/extension.pid
tail -f logs/extension.log
```

`selection.json` 为单种子验证筛选结果，`metrics_all.json` 为三个模型各三种子的逐条件测试记录，`summary.json` 为均值、样本标准差及配对差值。`evidence/` 保存各实际训练的验证摘要与复杂度。运行会写出逐图预测 `predictions.csv` 和 `original_predictions.csv` 以及 `results/` 下的检查点和训练历史。已发布的文件应在新的 Git 克隆副本里复现，以免覆盖原结果。自动筛选基于验证集，复跑的候选模型可能因数值差异而变化。第二轮复用第一轮划分和已观察过的测试集。原版、小图版、小图+SE 均为三种子 60 轮实验。小图+CBAM 只参加单种子验证筛选。
