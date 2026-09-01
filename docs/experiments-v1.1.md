# v1.1 实验结果

本报告记录冻结版 `v1.1-grasp-annotation` 的正式实验，不包含算法参数调优。
所有运行均使用 deterministic seed `0`、毫米单位、完整点云碰撞验证、闭合修正和
SE(3) 去重（平移 5 mm、旋转 10°）。

## 主实验：cone 模式

固定参数：`5 views × 3 anchors/view × 5 approaches`，cone angle `15°`，normal
KNN `30`，depth samples `16`。

| Object | Generated | Closure-valid | Unique | HQ≥0.8 | HQ/Unique | HQ/Generated | Top-20 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| juxing | 7,504 | 7,475 | 7,129 | 2,555 | 35.84% | 34.05% | 0.9899 |
| yuanzhu | 4,132 | 3,997 | 3,948 | 1,481 | 37.51% | 35.84% | 0.9999 |
| sanjiao | 8,757 | 8,599 | 8,412 | 2,368 | 28.15% | 27.04% | 0.9983 |
| huixing | 3,849 | 3,730 | 3,730 | 811 | 21.74% | 21.07% | 0.9851 |
| shuilongtou | 12,197 | 11,923 | 6,061 | 713 | 11.76% | 5.85% | 0.9966 |
| cat | 5,527 | 5,237 | 4,976 | 310 | 6.23% | 5.61% | 0.9753 |

原始 CSV：`results/ours-main/ours_main_results.csv`。

### 主实验 score 分布

对 34,256 个 closure-valid unique candidates 的加权统计如下；相对于 41,966 个
generated candidates 的 HQ yield 为 19.63%。

| 指标 | 数值 |
|---|---:|
| Mean / median | 0.3679 / 0.5795 |
| P10 / P25 / P75 / P90 / P95 | -0.8728 / 0.2378 / 0.7946 / 0.8865 / 0.9378 |
| Score ≥ 0 | 27,798 (81.15%) |
| Score ≥ 0.5 | 19,589 (57.18%) |
| Score ≥ 0.8 | 8,238 (24.05% of unique; 19.63% of generated) |
| Score ≥ 0.9 | 2,909 (8.49%) |
| Score ≥ 0.95 | 1,348 (3.94%) |

这里的 unique 数量表示几何上有效的候选集合，不等同于高质量标注数量；阈值统计可用于
后续构造 score-filtered annotation subset。完整逐对象数据见
`results/ours-main/ours_main_quality_stats.csv`，加权汇总见
`results/ours-main/ours_main_quality_aggregate.csv`。

## 消融实验

选择 `juxing`、`shuilongtou`、`cat`，统一使用 `3 views × 2 anchors/view`，仅改变
approach sampling mode：`global`、`normal`、`cone`。该设置用于控制消融运行时间，
不是主实验参数。

| Object | Mode | Generated | Closure-valid | Unique | HQ≥0.8 | HQ/Generated | Top-20 mean |
|---|---|---:|---:|---:|---:|---:|---:|
| juxing | global | 288 | 281 | 281 | 94 | 32.64% | 0.9750 |
| juxing | normal | 536 | 535 | 535 | 156 | 29.10% | 0.9666 |
| juxing | cone | 2,647 | 2,636 | 2,615 | 838 | 31.66% | 0.9687 |
| shuilongtou | global | 472 | 467 | 276 | 0 | 0.00% | 0.0290 |
| shuilongtou | normal | 949 | 920 | 491 | 37 | 3.90% | 0.9802 |
| shuilongtou | cone | 4,726 | 4,542 | 2,420 | 169 | 3.58% | 0.9961 |
| cat | global | 284 | 273 | 273 | 23 | 8.10% | 0.8724 |
| cat | normal | 459 | 430 | 430 | 21 | 4.58% | 0.8745 |
| cat | cone | 2,256 | 2,147 | 2,147 | 113 | 5.01% | 0.9367 |

原始 CSV：`results/ablation/ablation_results.csv`。

消融集合共包含 9,468 个 unique candidates。加权 score 阈值比例为：

| Score threshold | Count | Ratio |
|---:|---:|---:|
| ≥ 0 | 5,501 | 58.10% |
| ≥ 0.5 | 3,636 | 38.40% |
| ≥ 0.8 | 1,451 | 15.33% |
| ≥ 0.9 | 506 | 5.34% |
| ≥ 0.95 | 270 | 2.85% |

完整数据见 `results/ablation/ablation_quality_stats.csv` 和
`results/ablation/ablation_quality_aggregate.csv`。

本轮实验冻结 `HIGH_QUALITY_THRESHOLD = 0.8`。统计脚本默认使用该阈值，也支持通过
`--hq-threshold` 显式覆盖；正式主表和消融表均使用 `0.8`，避免事后挑选阈值。

## 可视化产物

无窗口环境下使用 Matplotlib 线框导出器生成每个对象的 Top-1/Top-20 PNG：

- 主实验：`results/ours-main/visualizations/`
- 消融实验：`results/ablation/visualizations/<object>_<mode>/`
- 高质量主实验（`score_total >= 0.8`）：`results/ours-main/visualizations-hq/`

导出命令示例：

```powershell
F:\Miniconda\envs\py310\python.exe scripts/export_experiment_visualizations.py `
  --object model\colmap\cat.ply `
  --results results\ours-main\cat `
  --output-dir results\ours-main\visualizations `
  --topk 20
```

论文用高质量线框图命令：

```powershell
F:\Miniconda\envs\py310\python.exe scripts/export_experiment_visualizations.py `
  --object model\colmap\cat.ply `
  --results results\ours-main\cat `
  --output-dir results\ours-main\visualizations-hq `
  --topk 20 --score-threshold 0.8 --dpi 300
```

## 可复现批处理

主实验和消融实验的运行日志分别位于 `results/ours-main/logs/` 和
`results/ablation/logs/`。结果目录包含 `grasps.json`、`grasps.npz` 和 `meta.json`，
其中 `meta.json` 保存配置、候选漏斗计数及各阶段耗时。

## Open3D 交互式线框查看

使用下面的入口会打开可交互的三维 Open3D 窗口，默认依次加载 6 个主实验模型。每个
模型按 `Top-1 → Top-20 → high-quality (score≥0.8)` 切换：

```powershell
F:\Miniconda\envs\py310\python.exe scripts/visualize_grasp_sequence.py
```

窗口操作：空格切换下一个状态，鼠标拖动旋转，滚轮缩放，`S` 保存当前视角截图，
`Q` 或 `Esc` 退出。截图保存到 `results/interactive-screenshots/`。也可以通过重复
`--model name=object_path=result_directory` 指定模型，例如：

```powershell
F:\Miniconda\envs\py310\python.exe scripts/visualize_grasp_sequence.py `
  --model cat=model\colmap\cat.ply=results\ours-main\cat
```
