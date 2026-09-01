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

## 评分诊断验证（不改核心算法）

对已有 `grasps.json` 离线计算接触带法向、双侧支撑、归一化中心距离和稳定性，
并将旧排名与诊断排名的 Top-20 进行比较。诊断结果显示旧排名的 Top-20 明显偏离
物体中心，而诊断排名降低了中心距离并提高了双侧法向一致性：

| Object | Old center/R | Diagnostic center/R | Old support | Diagnostic support | Old normal | Diagnostic normal |
|---|---:|---:|---:|---:|---:|---:|
| juxing | 0.4421 | 0.1412 | 0.0652 | 0.2661 | 0.7764 | 0.9841 |
| shuilongtou | 0.8951 | 0.1469 | 0.0259 | 0.0973 | 0.3574 | 0.9704 |
| cat | 0.6897 | 0.1515 | 0.0437 | 0.0549 | 0.5841 | 0.9294 |

这只验证 ranking diagnosis，不替换当前 `score_total`，也不重新生成 grasp。完整每个
grasp 的诊断字段位于本地 `results/ours-main/v4-diagnostics/`，汇总表为
`results/ours-main/v4_diagnostics_summary.csv`。

## Formal V4 离线重评分（不替换 V3 流水线）

为验证“旧评分导致边缘 Top-1”的诊断，新增 `grasp_score_v4.py`。它只读取已有
`grasps.json`，不重新生成候选、不修改碰撞和闭合修正，也不覆盖 `score_total`。
V4 使用 `2.5 * voxel_size` 接触带、双侧稳健（median）法向、左右支撑面积的几何均值，
以及以物体半径归一化的软中心性；最终分数为三个分量的无权几何均值：
`score_total_v4 = (score_v4_normal * score_v4_support * score_v4_stability) ** (1/3)`。
`score_v4_normal_dispersion` 是法向一致性分数（越大越一致），原 V3 分数保存在
`score_total_v3` 中。

重评分命令：

```powershell
F:\Miniconda\envs\py310\python.exe scripts/rescore_v4.py `
  --input-csv results\ours-main\v4_diagnostics_summary.csv `
  --output-dir results\ours-main\v4-rescore `
  --summary-csv results\ours-main\v4_rescore_summary.csv `
  --topk 20
```

本次六个对象的 V4 Top-20 汇总如下。该表用于确认排名趋势，V4 分数与 V3 分数不作
数值等价比较；在把 V4 接入生成流水线前仍需完成固定的 canonical grasp 验收。

| Object | V3 Top-20 center/R | V4 Top-20 center/R | V3 Top-20 support | V4 Top-20 support | V3 Top-20 normal | V4 Top-20 normal | Top-20 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| cat | 0.6897 | 0.2676 | 0.0254 | 0.0571 | 0.6321 | 0.8649 | 0 |
| huixing | 0.1993 | 0.2048 | 0.1276 | 0.2205 | 0.9925 | 0.9759 | 2 |
| juxing | 0.4421 | 0.1527 | 0.0312 | 0.2667 | 0.7233 | 0.9774 | 0 |
| sanjiao | 0.2878 | 0.2825 | 0.0102 | 0.1298 | 0.6561 | 0.8660 | 0 |
| shuilongtou | 0.8951 | 0.1670 | 0.0126 | 0.0848 | 0.6257 | 0.9867 | 0 |
| yuanzhu | 0.2527 | 0.3044 | 0.0510 | 0.2097 | 0.9650 | 0.8146 | 0 |

V4 Top-1 记录分别为 `cat=1588`、`huixing=21`、`juxing=4642`、`sanjiao=2365`、
`shuilongtou=3596`、`yuanzhu=517`。逐抓取结果位于
`results/ours-main/v4-rescore/`，汇总位于 `results/ours-main/v4_rescore_summary.csv`。

## V4 组件贡献与 canonical 排序验证

在冻结公式前，对六个模型的已有 V4 重评分结果进行了组件贡献、分布和排序检查：

```powershell
F:\Miniconda\envs\py310\python.exe scripts/analyze_v4_components.py `
  --input-dir results\ours-main\v4-rescore `
  --component-csv results\ours-main\v4_component_summary.csv `
  --distribution-csv results\ours-main\v4_score_distribution.csv

F:\Miniconda\envs\py310\python.exe scripts/validate_v4_ranking.py `
  --output-csv results\ours-main\v4_pairwise_checks.csv
```

四个 canonical pairwise case（中心大支撑、轻微偏心但双侧几何更好、大支撑对小支撑、
局部稳定柄部对中心弱支撑）全部通过。六个对象的 V4 Top-20 平均组件为：

| Object | Normal | Support | Stability | V4 total |
|---|---:|---:|---:|---:|
| cat | 0.8649 | 0.0571 | 0.8075 | 0.3365 |
| huixing | 0.9759 | 0.2205 | 0.8762 | 0.5714 |
| juxing | 0.9774 | 0.2667 | 0.9344 | 0.6243 |
| sanjiao | 0.8660 | 0.1298 | 0.7945 | 0.4446 |
| shuilongtou | 0.9867 | 0.0848 | 0.9121 | 0.4226 |
| yuanzhu | 0.8146 | 0.2097 | 0.7708 | 0.5058 |

组件与 V4 排名的 Spearman 相关在六个模型中均为负值，说明更高的法向、支撑和稳定性
总体都对应更靠前的结果；没有看到中心性单独决定全部排名的现象。完整组件汇总、分布和
pairwise 结果分别位于 `results/ours-main/v4_component_summary.csv`、
`results/ours-main/v4_score_distribution.csv` 和 `results/ours-main/v4_pairwise_checks.csv`。
该验证仍不接入主流水线，也不定义 V4 HQ 阈值。

## V4 主流程集成 smoke

V4 已正式放到闭合修正和后验证之后、SE(3) 合并之前。主流程固定为：

```text
Generate → V3 auxiliary score → Closure refine → Post-refinement validation
→ V4 score → V4-based symmetry-aware merge → Normalize/export
```

最终导出记录的 `score_total` 为 V4，旧分数保存在 `score_total_v3`；`meta.json` 增加
`score_version: "v4"`，NPZ 同时保存 `scores_total_v3`、`scores_total_v4` 和三个 V4 分量。

小规模真实对象 smoke（`1 view × 2 anchors × 5 approaches`）结果：

| Object | Raw after validation | Unique after V4 merge | Score version | Export |
|---|---:|---:|---|---|
| juxing | 1,028 | 1,028 | v4 | passed |
| shuilongtou | 1,524 | 790 | v4 | passed |
| cat | 632 | 632 | v4 | passed |

结果目录为 `results/v4-integration-smoke/<object>/`。这三组结果用于验证集成链路，
不覆盖原有 `v1.1` 实验目录；正式六对象重跑应在此 smoke 通过后单独执行。
