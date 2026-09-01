# V4 高质量抓取评价协议

该协议冻结 V4 评分的评价阈值，不改变抓取候选生成、碰撞检测、闭合修正或
SE(3) 去重算法。算法版本仍为 `v1.2-grasp-annotation`，既有 tag 不移动。

## 冻结配置

```text
score_version = v4
V4_HIGH_QUALITY_THRESHOLD = 0.13
threshold_source = manual_calibration
calibration_samples = 60 (G=19, B=25, U=16)
```

判定规则为 `HQ iff score_total_v4 >= 0.13`。不确定样本 `U` 只用于记录和
sanity check，不参与阈值拟合。

校准集的全局结果为 balanced accuracy `0.8674`、sensitivity `0.8947`、
specificity `0.8400`。留一对象验证阈值为 `cat=0.13`、`juxing=0.13`、
`shuilongtou=0.16`，因此正式协议采用全体校准集上 Balanced Accuracy 最大的
`0.13`，而不是对留一结果取平均。

邻域 sanity check（同一 60 条标签集）：

| threshold | balanced accuracy | sensitivity | specificity |
|---:|---:|---:|---:|
| 0.11 | 0.8274 | 0.8947 | 0.7600 |
| 0.12 | 0.8274 | 0.8947 | 0.7600 |
| **0.13** | **0.8674** | **0.8947** | **0.8400** |
| 0.14 | 0.8411 | 0.8421 | 0.8400 |
| 0.15 | 0.8147 | 0.7895 | 0.8400 |
| 0.16 | 0.8147 | 0.7895 | 0.8400 |
| 0.17 | 0.7884 | 0.7368 | 0.8400 |

`U` 样本的 V4 分数范围为 `[0.0000, 0.4517]`，median `0.1005`，P25
`0.0000`，P75 `0.2627`。它们不重新加入 G/B 校准；该分布仅作为人工边界
不确定性的记录。

## 正式实验参数

六个对象均使用同一套 deterministic 参数，并按顺序串行运行：

```text
algorithm = v1.2-grasp-annotation
mode = cone
views = 5
anchors/view = 3
approaches/anchor = 5
cone = 15 degrees
normal KNN = 30
depth samples = 16
seed = 0
merge = 5 mm / 10 degrees
HQ threshold = 0.13
```

运行示例：

```powershell
F:\Miniconda\envs\py310\python.exe main.py `
  --object model\juxing.ply `
  --views 5 --anchors 3 --mode cone `
  --output results\formal-v1.2\juxing
```

每个结果目录包含 `grasps.json`、`grasps.npz` 和 `meta.json`；其中 `meta.json`
记录阈值、来源、校准样本数、HQ 数量及运行参数。主表使用：

```text
HQ/Unique = high_quality_count / unique_grasp_count
HQ Yield   = high_quality_count / generated_candidate_count
```

消融实验仍只改变 `global / normal / cone` 的 approach sampling，其余评分、
碰撞、闭合修正、阈值和去重全部保持一致。
