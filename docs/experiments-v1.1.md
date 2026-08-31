# v1.1 实验结果

本报告记录冻结版 `v1.1-grasp-annotation` 的正式实验，不包含算法参数调优。
所有运行均使用 deterministic seed `0`、毫米单位、完整点云碰撞验证、闭合修正和
SE(3) 去重（平移 5 mm、旋转 10°）。

## 主实验：cone 模式

固定参数：`5 views × 3 anchors/view × 5 approaches`，cone angle `15°`，normal
KNN `30`，depth samples `16`。

| Object | Generated | Closure-valid | Unique | Closure rate | Merge retention | Mean score | Top-1 | Top-20 mean | Total (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| juxing | 7,504 | 7,475 | 7,129 | 99.61% | 95.37% | 0.6913 | 0.9953 | 0.9899 | 597.21 |
| yuanzhu | 4,132 | 3,997 | 3,948 | 96.73% | 98.77% | 0.6864 | 1.0000 | 0.9999 | 565.82 |
| sanjiao | 8,757 | 8,599 | 8,412 | 98.20% | 97.83% | 0.6095 | 0.9996 | 0.9983 | 1,053.92 |
| huixing | 3,849 | 3,730 | 3,730 | 96.91% | 100.00% | 0.5776 | 0.9902 | 0.9851 | 606.17 |
| shuilongtou | 12,197 | 11,923 | 6,061 | 97.75% | 50.83% | -0.5638 | 0.9966 | 0.9966 | 870.24 |
| cat | 5,527 | 5,237 | 4,976 | 94.75% | 95.02% | 0.2214 | 0.9781 | 0.9753 | 832.88 |

原始 CSV：`results/ours-main/ours_main_results.csv`。

## 消融实验

选择 `juxing`、`shuilongtou`、`cat`，统一使用 `3 views × 2 anchors/view`，仅改变
approach sampling mode：`global`、`normal`、`cone`。该设置用于控制消融运行时间，
不是主实验参数。

| Object | Mode | Generated | Closure-valid | Unique | Top-1 | Top-20 mean | Total (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| juxing | global | 288 | 281 | 281 | 0.9950 | 0.9750 | 20.41 |
| juxing | normal | 536 | 535 | 535 | 0.9673 | 0.9666 | 38.56 |
| juxing | cone | 2,647 | 2,636 | 2,615 | 0.9712 | 0.9687 | 235.11 |
| shuilongtou | global | 472 | 467 | 276 | 0.4305 | 0.0290 | 26.40 |
| shuilongtou | normal | 949 | 920 | 491 | 0.9964 | 0.9802 | 63.02 |
| shuilongtou | cone | 4,726 | 4,542 | 2,420 | 0.9964 | 0.9961 | 365.83 |
| cat | global | 284 | 273 | 273 | 0.9321 | 0.8724 | 35.49 |
| cat | normal | 459 | 430 | 430 | 0.9573 | 0.8745 | 61.14 |
| cat | cone | 2,256 | 2,147 | 2,147 | 0.9822 | 0.9367 | 342.03 |

原始 CSV：`results/ablation/ablation_results.csv`。

## 可视化产物

无窗口环境下使用 Matplotlib 线框导出器生成每个对象的 Top-1/Top-20 PNG：

- 主实验：`results/ours-main/visualizations/`
- 消融实验：`results/ablation/visualizations/<object>_<mode>/`

导出命令示例：

```powershell
F:\Miniconda\envs\py310\python.exe scripts/export_experiment_visualizations.py `
  --object model\colmap\cat.ply `
  --results results\ours-main\cat `
  --output-dir results\ours-main\visualizations `
  --topk 20
```

## 可复现批处理

主实验和消融实验的运行日志分别位于 `results/ours-main/logs/` 和
`results/ablation/logs/`。结果目录包含 `grasps.json`、`grasps.npz` 和 `meta.json`，
其中 `meta.json` 保存配置、候选漏斗计数及各阶段耗时。
