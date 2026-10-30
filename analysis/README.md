# analysis/ — 诊断与对照实验脚本（2026-10-28）

这些脚本独立于 `ehr_forecast/`，只读原始 CSV 和 `artifacts/optimization/prepared_l512`，不修改任何已有产物。
结论见 `docs/diagnosis_and_improvement_plan.md`。运行方式（项目根目录）：

```powershell
cd analysis
..\.venv\Scripts\python.exe ehr_load.py            # 建缓存 ehr_cache.pkl（锚点、标签、事件长表；标签与项目 audit.json 逐一核对一致）
..\.venv\Scripts\python.exe analyze_dgp.py         # 生成过程诊断：死亡/锚点结构、前驱提升、数值观测 AUROC、时间到事件
..\.venv\Scripts\python.exe tabular_baseline.py    # 计数特征 + LightGBM / LogReg，val 与 train 5 折 OOF
..\.venv\Scripts\python.exe cv_all.py              # val 是否比 train 偏易：3156 人 5 折 CV，分子集报告
..\.venv\Scripts\python.exe tune_gbm.py            # GBM 配置网格（OOF 判定）+ 重复目标平均
..\.venv\Scripts\python.exe augment_landmarks.py   # 多 landmark 增强（GBM）+ 死亡可预测性 / 年代代理
..\.venv\Scripts\python.exe dense_transformer.py sparse|dense|dense_mix 17,42,73   # 稀疏 vs 密集监督（GPU）
..\.venv\Scripts\python.exe aux_transformer.py sparse|multi|wide|multi_wide 17,42,73  # 辅助头 / wide 通路（GPU）
..\.venv\Scripts\python.exe blend.py               # 各模型在 val 上的融合（仅 val，含选择乐观）
..\.venv\Scripts\python.exe pipeline_v2.py         # 迭代 2：OOF 框架 + 更丰富特征 + 两阶段 + 收缩 → predictions_v2.csv
..\.venv\Scripts\python.exe pipeline_v3.py         # 迭代 3：竞争风险通路 softmax → predictions_v3.csv
..\.venv\Scripts\python.exe cv_transformer.py multi_wide 17,42   # 迭代 4：Transformer 5 折 OOF（GPU）
..\.venv\Scripts\python.exe final_blend.py v3 multi_wide         # 在 OOF 上选融合权重 → predictions_final.csv
```

`v2/`、`v3/`、`cv_tf/` 保存各迭代的 OOF / val / test 概率；`per_condition_final.csv`、`summary_final.json` 为最终结果。

数据审计与干净数据集（见 `docs/data_audit.md`）：

```powershell
..\.venv\Scripts\python.exe data_audit_a.py        # 队列/划分偏移/各表/时间结构/年代标记/数值单位
..\.venv\Scripts\python.exe data_audit_b.py        # 基因组、DICOM、REASONCODE、未来信息通道清单
..\.venv\Scripts\python.exe build_clean_dataset.py [--no-qol]   # 干净特征矩阵 clean/X.parquet + manifest.json
..\.venv\Scripts\python.exe pipeline_clean.py      # 干净数据上的 OOF 基线 → predictions_clean.csv
..\.venv\Scripts\python.exe information_ceiling.py # 信息天花板：年龄性别 / +10 前驱 / GBM 的 OOF 对比
..\.venv\Scripts\python.exe cv_transformer_hybrid.py 17,42   # 全史 wide 通路的混合 Transformer 5 折 OOF（GPU）
..\.venv\Scripts\python.exe final_blend.py clean hybrid      # 干净表格 + 混合 Transformer，OOF 选权重
..\.venv\Scripts\python.exe person_period.py                 # person-period 风险模型（30 非终末目标，XGBoost-GPU）；_mcw1 / _v2 / _lgb 为变体
..\.venv\Scripts\python.exe sparse_additive_gpu.py           # 批量 L1 逻辑回归（GPU）+ XGB stumps
..\.venv\Scripts\python.exe tune_v4.py / tune_v5.py          # 筛选、bagging、GPU 随机森林、逐目标超参（rep0 选 / rep1 评）
..\.venv\Scripts\python.exe lgb_leaf_sweep.py                # LightGBM 叶子约束扫描（CPU）
..\.venv\Scripts\python.exe assemble_final.py                # 逐目标贪心集成选择（交叉拟合 OOF）→ predictions_final.csv
```

第五轮迭代（2026-10-29，见 `docs/diagnosis_and_improvement_plan.md` §10.2）：

```powershell
..\.venv\Scripts\python.exe stack_l2.py                     # 二层 stacking（各族 OOF logit + 交叉拟合死亡概率 + 跨目标）→ stack/
..\.venv\Scripts\python.exe pp_offset_gbm.py pp2 pp_yearly_uniform_   # offset boosting（负结果）→ offset/
..\.venv\Scripts\python.exe mechanism_probe.py              # 机制探针：单变量 AP、目标专属子集 GBM vs 全特征
..\.venv\Scripts\python.exe block_gbm.py                    # 官方行嵌套特征块 S1/S2/S3 LightGBM → blocks/
..\.venv\Scripts\python.exe pp_build.py [--step 2] [--transductive --out ppds_tr]   # 缓存 person-period 数据集（含 is_official / yrs_to_end 指示）→ ppds/
..\.venv\Scripts\python.exe pp_sweep2.py d3,d6 --targets all [--block S1|S2|S3] [--ds ppds_tr]   # XGBoost-GPU 扫描，每折共享 QuantileDMatrix
..\.venv\Scripts\python.exe pp_sweep_lgb.py nl8 --block all [--seed 1]   # 同一数据集的 LightGBM（CPU）版本 → pplgb/
..\.venv\Scripts\python.exe assemble_final.py --rank --cf-reps 5 --only "<候选名列表>"   # 逐目标贪心集成（秩平均、交叉拟合估计多次对半平均）→ predictions_final.csv
```

`logs/` 保存了本次运行的全部输出；`*_val.npy` 为各模型的 val 概率矩阵（365×40，列序同 target_conditions.csv）。
所有 Transformer 对照实验共用 `prepared_l512` 的 token 与词表；时间特征改为"事件时年龄 + 距上一事件间隔 + 性别"，并在官方 anchor 处追加一个 [PRED] token（id 3），以保证逐位置标签不泄漏 anchor 时间。

## 第六轮（2026-10-29）：上限 / 学习曲线 / 样本效率诊断（文档 §10.3）

- `oracle_ceiling.py` — 官方行 LightGBM：随机基线（流行率）、窗口内死亡 / 死亡时刻 / 窗口内其他事件 oracle、训练患者 25/50/75% 学习曲线 → `oracle/`
- `pp_sweep2.py d3 --frac 0.25|0.5|0.75` — person-period 学习曲线（`ppds/d3_all_f*.npy`）
- `ap_consistency.py` — Platt / isotonic 校准后"按自身风险抽签"的期望 AP vs 实际 AP，两点混合闭式例 → `oracle/ap_consistency.csv`
- `death_mixture.py` — 官方行：死亡 0/1 真值特征、交叉拟合 P(死亡) 特征、死亡门控混合专家（门控 = P(死亡) / 真值）→ `mix/`（装配候选布局）
- `pp_mix.py --mode mix|pdfeat` — person-period 版混合专家 / logit P(死亡) 特征 → `ppds/mix_d3_all_*`, `ppds/pdfeat_d3_all_*`
- `screen_gbm.py 60,200` — 折内逐目标单变量 AP 特征筛选 + LightGBM → `screen/`（装配候选）
- `pp_screen.py 200,400` — 同样的筛选用在 person-period XGB d3 上 → `ppds/screen{K}_d3_all_*`
- 逐目标学习曲线汇总（官方行 + person-period 两条曲线的 AP 与斜率，§10.3 补记 2）→ `oracle/per_condition_learning_curves.csv`
- `assemble_final.py` 现在也自动发现 `screen/`、`mix/` 候选；探测日志 `logs/assemble_probe_r6*.log`

## 第七轮（2026-10-29）：按 README 许可加入 genomics / imaging（文档 §10.4）

- `build_genomics_features.py` — `genomics/*_dna.csv`（161 rsID 剂量、基因计数、变异计数）+ `imaging/*.dcm`（仅 StudyDate ≤ anchor 的计为 `has_dicom_pre`）→ `clean/genomics_features.parquet`
- `genfeat.py` — `EHR_GENOMICS=1` 时把上述特征并入 `clean/X.parquet` 并把输出目录改为 `screen_g/`、`blocks_g/`、`stack_g/`、`v4_g/`（已接入 `screen_gbm.py`、`block_gbm.py`、`stack_l2.py`、`sparse_additive_gpu.py`）
- `ppds_g/` — 含 genomics 列的 person-period 数据集；`pp_sweep2.py d3 --ds ppds_g`、`pp_screen.py 200 --ds ppds_g`、`pp_sweep_lgb.py nl8 --ds ppds_g --block S1 --tag _g`
- `gen_stack.py` — 最终装配分 + 基因组特征的逐目标第二级 → `gstack/`
- `assemble_final.py --rank --cf-reps 5 --only "$(cat final/curated_candidates.txt)"` → 根目录 `predictions.csv`（40 候选，OOF 0.2671 / val 0.3158）；历史提交在 `final/archive/`

## 第八轮（2026-10-30）：费用字段 + 特征分组地图 + 两阶段残差（文档 §10.5）

- `build_cost_features.py` — patients.csv 终生费用两列 vs 事件表 anchor 前费用之和（恒等式检验 + 分解）→ `clean/cost_features.parquet`（13 列）
- `build_cost_features_v2.py` — + anchor 前 1 / 5 年支出窗口 + 交叉拟合 LightGBM 回归的两阶段残差 + 残差 × P(死亡) → `clean/cost_features_v2.parquet`（25 列）
- `feature_group_map.py` — 每个目标在各特征组内的最佳单变量 AUROC，费用信号按窗口内生死分层 → `oracle/feature_group_map.csv`
- `genfeat.py` — `EHR_GENOMICS=2`（基因组 + 费用 v1 → `*_gc/`）、`=3`（基因组 + 费用 v2 → `*_gcr/`）
- `ppds_gc/`、`ppds_gcr/` — 含费用 v1 / v2 列的 person-period 数据集；`pp_sweep2.py d3 --ds ppds_gc|ppds_gcr`、`pp_screen.py 200 --ds …`、`pp_mix.py --mode pdfeat|mix --ds …`、`pp_sweep_lgb.py nl8 --ds … --block S1 --tag _gc|_gcr`
- `gen_only.py [cost|cost2]`、`gen_stack.py [cost|cost2]` — 无历史模型与第二级，费用 v1 / v2 版本 → `gstack/genc_only`, `genc2_only`, `final7+genc`, `final7+genc2`
- `assemble_final.py` 自动发现 `*_gc/`、`*_gcr/`；候选清单 `final/curated_candidates.txt`（当前提交）、历史清单与提交在 `final/archive/`；探测日志 `logs/assemble_probe_gc*.log`, `assemble_probe_gcr_p*.log`

## 第九轮（2026-10-30）：训练量 / 容量 / 神经网络 / 集成层 / 费用 v3（文档 §10.6）

- `rounds_curve.py` — 官方行 LightGBM 50–1200 轮的 OOF 曲线 → `oracle/rounds_curve_lgb_gcr3.csv`
- `mlp_multitask.py off|pp` — 多任务表格 MLP（共享主干 + 40 头），官方行 / person-period 行 → `nn/mlp_{off,pp}_*`
- `nn_variants.py ple|resid` — 分段线性编码 MLP；以 `ppds_gcr:screen200` logit 为固定偏置的残差网络 → `nn/{ple,resid}_*`
- `assemble_final.py --shrink α [--shrink-level group|module]` — 层次化集成权重（为负，默认关闭）
- `build_cost_features_v3.py` — 费用 v3（+ 保险方构成）→ `clean/cost_features_v3.parquet`；`genfeat.py` 模式 4 → `*_gcr3/`；`ppds_gcr3/`
- `screen_gbm.py`（`LGB_N` 环境变量改轮数）、`pp_screen.py`（`PP_ALWAYS_DENSE=1` 强制保留稠密列）、`pp_sweep2.py` 新配置 `d3n200`
- 探测日志 `logs/assemble_probe_r9_*.log`、`assemble_shrink_*.log`；提交未变（`predictions.csv` = 第八轮 41 候选）

## 第十轮（2026-10-30）：数据逐列清点（文档 §10.7）

- test 六张表 STOP 截断核对、medications DISPENSES 隐含时长（`clean/med_duration_features.parquet`，无信号）、地理 / 婚姻字段、稀有编码数量、终末组死者内部特征地图、P(死亡)^γ 事后重加权——均无增益；提交不变

## 第十一轮（2026-10-30）：紧凑模型 / 正例加权 / 列采样 / 装配器设置（文档 §10.8）

- `screen_gbm.py 20,40`（`EHR_GENOMICS=3`）、`pp_screen.py 50 --ds ppds_gcr`、`pp_sweep2.py d3spw3,d3cs6 --ds ppds_gcr`、`assemble_final.py --steps 16|4 / --bagged`——全部在 ±0.002 内；提交不变

## 第十二轮（2026-10-30）：按设计优化 3 成员（文档 §10.10）

- `pp_screen.py --seed N`、`screen_gbm.py`（`LGB_SEED`、`LGB_N_GROUP=acc:100,chain:150,term:350`）、`gen_only.py cost2 seedN`、`avg_candidates.py`（种子平均）
- `gate_combiner.py [--lr] <out> <dir:stem>...` — 显式死亡门控合并器（LightGBM / 逻辑回归）→ `gstack/`
- `temporal_transformer.py [maxlen] [seeds]` — 时间 Transformer over 每年状态向量 → `nn/tts_*`
- 组合规则对照：`assemble_final.py` 的 `--rank` / 无 / `--shrink` / `--steps`；全部在 ±0.0013 内，提交不变
