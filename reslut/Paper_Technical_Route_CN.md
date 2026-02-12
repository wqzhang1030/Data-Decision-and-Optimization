# 基于病理 Patch–Patient 两阶段学习的幽门螺杆菌相关诊断研究

## 摘要
本文围绕幽门螺杆菌相关病理图像分析任务，构建并实现了一条从 patch 级别到 patient 级别的两阶段技术路线。第一阶段在 `Presence ∈ {1, -1}` 的 patch 标签上训练二分类模型，第二阶段将 patch 模型输出与深层特征按患者聚合后进行三分类（`NEGATIVA / BAIXA / ALTA`）。

在严格患者分组（grouped by `Pat_ID`）和 HoldOut 最终评估设置下，本研究保留的最佳 patient 模型在 HoldOut 上达到：`Accuracy=0.7759`、`Macro-F1=0.7492`、`Balanced-Accuracy=0.7368`、`Macro OvR ROC-AUC=0.7978`。其中 BAIXA 类别达到 `Precision=0.6250`、`Recall=0.5882`、`F1=0.6061`。此外，本文还生成了 patch 级 Grad-CAM 热力图与患者级预测对照 xlsx，用于可解释性与误差追踪。

---

## 1. 研究目标与问题定义

### 1.1 任务目标
本项目目标是构建一条可复现、可解释的病理图像分析流程：

1. **Patch 级任务（Task1）**：预测每个 patch 的 `Presence`（阳性/阴性）。
2. **Patient 级任务（Task2）**：基于同一患者的 patch 信息，预测患者级 `DENSITAT` 三分类标签。

### 1.2 标签体系
- Patch 标签：来自 xlsx 的 `Presence ∈ {1, -1}`；`Presence == 0` 视为无效标签并丢弃。
- Patient 标签：来自 `PatientDiagnosis.csv` 的 `DENSITAT ∈ {NEGATIVA, BAIXA, ALTA}`。

---

## 2. 数据与映射规则

### 2.1 数据来源
- 图像目录：`/hhome/ricse03/HelicoData/CrossValidation/Annotated`
- Patch 标注表：`/hhome/ricse03/HelicoData/HP_WSI-CoordAnnotatedAllPatches.xlsx`
- 患者诊断表：`/hhome/ricse03/HelicoData/PatientDiagnosis.csv`

### 2.2 关键映射规则（工程硬约束）
1. `Pat_ID = patient_folder.split('_')[0]`
2. `Window_ID` 由图像文件 stem 与 xlsx `Window_ID` 进行规范化匹配
3. patch 标签只使用 `Presence ∈ {1, -1}`，忽略 `Presence == 0`

### 2.3 数据清洗与样本保留
在 clean rebuild 流程中：
- xlsx 总行数：`2695`
- 保留 patch：`2676`
- 丢弃项：
  - `Presence` 非 `{1,-1}`：`4`
  - 无匹配 PNG：`15`

患者标签总体分布（`patient_labels.csv`）：
- `NEGATIVA: 151`
- `ALTA: 86`
- `BAIXA: 72`

---

## 3. 整体技术路线

### 3.1 两阶段架构

**阶段A：Patch Presence 二分类**
- Backbone：ResNet 系列（保留最佳：`resnet18_s42`）
- 输出：每个 patch 的概率 `p_pos`

**阶段B：Patient 三分类**
- 输入：同一患者下所有 patch 的概率统计 + embedding 聚合特征
- 分类器：CV 选择的树模型（最佳 run 选中 `et_baixa_boost`）
- 输出：患者级 `pred_class` 与三类概率

### 3.2 防泄漏策略
- train/val/test 全部 **按患者分组** 切分（同一患者不跨集合）
- HoldOut 作为最终评估集，不参与训练与常规调参流程

---

## 4. 实现细节

### 4.1 Patch 级实现
- 关键脚本：`src/train/train_patch_presence.py`
- 模型定义：`src/models/resnet_presence.py`
- 最佳 checkpoint：`outputs/patch_matrix/resnet18_s42/run_001/best.ckpt`

Patch 集合规模（clean split）：
- Train：`1676` patches / `107` patients
- Val：`305` patches / `23` patients
- Test：`695` patches / `24` patients

标签分布（Presence）：
- Train：`+1:1086, -1:590`
- Val：`+1:221, -1:84`
- Test：`+1:153, -1:542`

### 4.2 Patient 特征构建
- 关键脚本：`src/infer/build_patient_embedding_features.py`
- 特征来源：
  1. patch 概率统计特征（均值、方差、分位数、top-k 等）
  2. patch 深层 embedding 的患者级聚合（均值/标准差）
- 该最佳设置下输入特征维度：`1033`

### 4.3 Patient 分类与评估
- 关键脚本：`src/train/train_patient_classifier_embed_cv.py`
- 策略：
  - 候选模型 CV 选择
  - BAIXA 导向阈值与模型搜索（权衡 macro-F1 与 BAIXA recall）
- 关键结果目录：`outputs/patient_full/resnet18_s42/run_001/`

### 4.4 可解释性与报告
- Patch Grad-CAM：`src/infer/generate_patch_gradcam_heatmaps.py`
  - 输出：`outputs/patch_heatmaps/run_001/`
  - 本次生成：`50` 张（`25` 阳性 + `25` 阴性）
- 患者级对照表：`src/report/export_patient_prediction_report.py`
  - 输出：`patient_prediction_report.xlsx`
  - 含真值、预测、正确性、置信度、错误样本表

---

## 5. 实验结果

### 5.1 Patch 级结果（最佳 patch run）
来源：`outputs/patch_matrix/resnet18_s42/run_001/metrics.json`

- Val：
  - Acc `0.9804`
  - F1 `0.9853`
  - ROC-AUC `0.9947`
  - PR-AUC `0.9977`
- Test：
  - Acc `0.9505`
  - F1 `0.9499`
  - ROC-AUC `0.9792`
  - PR-AUC `0.9863`

### 5.2 Patient 级结果（最佳 patient run）
来源：`outputs/patient_full/resnet18_s42/run_001/metrics.json`

- Accuracy：`0.7759`
- Macro-F1：`0.7492`
- Macro-Precision：`0.7677`
- Macro-Recall：`0.7368`
- Balanced-Accuracy：`0.7368`
- Macro OvR ROC-AUC：`0.7978`
- Macro OvR PR-AUC：`0.6709`
- Brier：`0.4384`
- LogLoss：`2.5991`
- ECE：`0.1296`

每类指标：
- NEGATIVA：P `0.8281` / R `0.9138` / F1 `0.8689`
- BAIXA：P `0.6250` / R `0.5882` / F1 `0.6061`
- ALTA：P `0.8500` / R `0.7083` / F1 `0.7727`

混淆矩阵：
- `[[53, 5, 0], [11, 20, 3], [0, 7, 17]]`

### 5.3 与早期版本对比（BAIXA 改善）
与历史基线（如 `outputs/patient/run_009`）相比，BAIXA recall 从约 `0.235` 提升到 `0.588`，同时整体 macro-F1 提升到 `0.749`，说明“分层患者切分 + BAIXA导向模型选择”对中间类识别具有显著作用。

---

## 6. 误差分析与可解释性观察

### 6.1 错误概况
在 HoldOut `116` 名患者中，错误 `26` 例（约 `22.4%`）。

从错误表（`patient_prediction_report.xlsx` 与 `paper_error_cases_top.csv`）可见：
- 高置信错误主要集中在 BAIXA 与 NEGATIVA / ALTA 的混淆。
- 说明 BAIXA 在特征空间中仍存在“中间态重叠”。

### 6.2 Grad-CAM 观察
通过 `outputs/patch_heatmaps/run_001/`：
- 阳性/阴性样本均有可视化覆盖。
- 热力图可用于辅助确认模型关注区域是否与组织学线索一致。

---

## 7. 产物与复现文件

### 7.1 关键结果文件
- Patch 最佳：`outputs/patch_matrix/resnet18_s42/run_001/`
- Patient 最佳：`outputs/patient_full/resnet18_s42/run_001/`
- 汇总：`outputs/patient_full/summary_metrics.csv`、`outputs/patient_full/summary_top.json`
- 可解释性：`outputs/patch_heatmaps/run_001/`
- 论文表格包：`outputs/paper/run_001/`

### 7.2 打包目录（报告交付）
- `/hhome/ricse03/reslut`
- `/hhome/ricse03/reslut/paper`

---

## 8. 结论与后续方向

本文完成了从 patch 到 patient 的完整工程闭环，并在患者三分类上取得 `Accuracy 0.7759 / Macro-F1 0.7492` 的结果。当前短板集中在 BAIXA 与相邻类别边界的判别稳定性。

下一步可重点推进：
1. 基于错误高置信样本做 hard-example 重加权训练
2. 引入 stain normalization 与更强 domain augmentation
3. 在患者层尝试序列/注意力式 bag 建模（替代纯统计聚合）
4. 使用外部独立队列进行泛化验证

---

## 附录：本稿对应实现文件

- 数据与清洗：`src/data/rebuild_clean_patch_splits.py`
- Patch 训练：`src/train/train_patch_presence.py`
- Embedding 特征：`src/infer/build_patient_embedding_features.py`
- Patient 训练：`src/train/train_patient_classifier_embed_cv.py`
- Patch 热力图：`src/infer/generate_patch_gradcam_heatmaps.py`
- 患者 xlsx：`src/report/export_patient_prediction_report.py`
- 论文产物构建：`src/report/build_paper_artifacts.py`

