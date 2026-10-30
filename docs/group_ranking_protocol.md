# 全局分组排序研究协议（2026-10-28）

目标是检验同病种患者组的排序监督能否改善单模型，而非继续更换病史编码结构。先完成输出诊断，再比较经典pairwise/listwise方法与GRPO启发的组内奖励。所有方法均是待验证假设，不承诺涨分。

## 已有证据和问题定义

主诊断对比为严格实验C1R0−C0R0：mAP −0.018435，Brier −0.001727，at-risk BCE −0.008611。Brier改善约98.6%来自负例项，BCE改善约93.8%来自正例项；不同checkpoint选中阶段也造成BCE表象反转。肺癌相关目标、脑微出血/步态异常的少数高位阳性被阴性超过；新读出的下降位置则主要是前列腺/肥胖。诊断详见 `experimental_audit_results.md` 和 `artifacts/ranking_diagnosis/`。

单调概率校准不能恢复被打乱的病种内排序。BCE/Brier和AP不构成理论上必然冲突；本实验要检验有限表征和训练下，显式的跨患者组排序目标是否有帮助。

## 固定数据、表示和可复现性

- 使用原官方split、prepared_l512、BOS+511事件、相同目标/既往病屏蔽；不使用外部数据或权重。
- 固定种子17/42/73，分别冻结对应严格C0R0 repeat-a的encoder。每位患者特征为last-state128维+原static5维，缓存train/val/test；新训练只更新同一个133→40 linear分类头。
- 缓存前验证原源码/数据/权重SHA；eval、float32、strict deterministic、不使用AMP。缓存头的初始验证预测必须与原保存预测最大差≤1e-6、mAP一致，且与完整forward一致。
- 所有方法同一种子共享特征和原头初始化；原模型未变，新增源码位于独立 `ranking_study/`，不修改原冻结模块。
- 所有目标函数只读取训练标签。验证患者与本文定位的错误病例不参与训练配对、采样、奖励、权重或温度拟合。

## 六组方法

对疾病k，P_k为训练集中未来新发阳性、N_k为anchor前未确诊且未来窗内无新发的阴性；已确诊患者排除。每个病种作为一个query、其全部eligible训练患者作为一个列表。每类损失先归一化，再40类等权。

1. **bce**：macro-BCE头再优化，作为所有排序方法的主要程序对照。
2. **ranknet**：macro-BCE + 正负score差的logistic pair损失。它监督pair顺序，不称为AP loss。
3. **listnet**：macro-BCE + positive-normalized listwise softmax交叉熵，目标在阳性患者间均匀分配质量。受ListNet启发，binary目标分布适配并非原论文公式，也不等同AP。
4. **smoothap**：macro-BCE + 1−平滑AP。用sigmoid近似正例阈值处的排序计数，全部训练列表计算，不用随机小batch AP代替全局目标。
5. **rloo_ap**：macro-BCE + 新鲜on-policy Plackett–Luce排列的AP策略梯度；奖励baseline为同病种其他G−1个排列的平均奖励。
6. **grpo_ap**：与rloo_ap相同的策略/采样/奖励，但优势改为同组奖励中心化并除以组标准差。称“GRPO启发的on-policy组相对AP策略梯度”，不称原版GRPO复现。

后二者“组”的含义是：同一疾病、同一批全部eligible训练患者，独立采G=8个排列；每个排列是一个action，完整AP是reward。不能把不同患者的0/1标签中心化后当作GRPO的排列奖励。

PL策略为每个剩余列表上softmax(score/temperature)连续选取下一位。使用Gumbel扰动采样，精确计算所采排列的log probability。每次Adam更新都重新采样，不复用旧轨迹，不加入没有必要的PPO ratio/clipping或critic。共同BCE与参数锚定用于约束偏移，不冒称原GRPO的参考策略KL。

RLOO的baseline独立于本次action，可作无偏策略梯度控制变量；除以同组随机reward标准差的GRPO变体一般会改变有限样本估计与病种权重，不宣称它就是raw expected macro-AP的无偏梯度。零方差组记录并贡献零梯度，不编造信号。

## 共同优化设置

- 全训练特征、float32 CUDA、严格确定性，Adam lr=0.003、weight_decay=0。
- 总目标为macro-BCE + lambda×排序项 + 0.1×ridge；ridge为0.5×每类参数相对原头的平方偏移和，再对40类取均值。BCE对照使用相同ridge。
- 温度逐病种固定为 `clamp(0.5 * std(initial eligible train logits, unbiased=False), 0.25, 2.0)`；不能用验证排名调温度。
- lambda在初始化时仅用训练梯度定标：`0.25 * ||grad(BCE)|| / ||grad(rank)||`，之后固定。RL使用4次独立G=8采样的平均梯度定标；梯度退化或非有限时停止该arm并记录，不静默退回BCE。没有验证集lambda搜索。
- calibration RNG为seed+27041，RL training RNG为seed+17041，独立于初始化。两个RL方法使用相同规则和随机流。其他方法确定性计算全局目标。
- 最多100个head epoch，每epoch完整评估；patience15，统一按验证mAP选择。epoch0原头也参与best，明确记录是否选择了epoch0；回退原头不算改进。
- 相同步数epoch10作为预先指定的附加诊断。所有方法均有相同选模机会，不拿一个方法的best与另一个方法的末轮相比。
- 记录macro-BCE（实际目标）和旧at-risk全局BCE（用于纵向比较），避免混淆。

冻结独立线性头后，各类目标不共享可训练参数。正常数类权重在无正则且完全收敛时不改变每类最优解；有限步数及全局正则缩放则可能改变结果。因此所有新方法共享相同逐类BCE和ridge归一化，以BCE再优化作主要对照，不能把相对原checkpoint的全部变化归为排序收益。

## 门禁和分析

1. 特征重载闭环与三种子输入/初始头校验。
2. CPU数学单元测试：PL概率和为1、logprob、AP奖励、RLOO小排列精确期望梯度、零方差、prior排除、平滑AP低温、梯度与有限值。
3. 完成6方法×3种子共18次head拟合，不根据前几个分数取消或追加某个方法。
4. 另将bce-seed17和grpo_ap-seed17完整重复各一次，只检验同环境可复现性，不用于选分。逐轮非时间指标、最佳头权重和预测应完全一致；失败先停止归因，不放宽容差。
5. 主比较为各方法相对同预算bce的单模型mAP逐seed差再均值/SD；使用共同患者bootstrap2000次，非概率集成。报告全部病种、固定长/短组、Brier/BCE、reward方差、epoch0回退和epoch10。
6. 优先判断是否超过bce再优化，而非仅超过原头。所有区间条件于已训练模型和反复使用的验证集，不能称无偏测试收益；多方法比较的选择偏差需显式保留。

最优方法只可按三种子平均验证mAP选择实验候选，不能挑种子。若需要一个可运行候选，预先固定seed42作为演示权重；三份单模型仍全部保留，不做集成。若不超过BCE对照，结论写未改善，不继续搜索直到涨分。根目录提交预测和此前模型包本阶段不替换。

## 数学意义与边界

固定features和温度后，BCE+RankNet+非负二次ridge对linear头是凸问题；ListNet适配也为凸目标。SmoothAP与AP策略梯度不保证凸。各类全局正负pair总数10,725,270、最大单类882,954，缓存表示后可以分块精确计算，不需要把2791人全部送进Transformer反传。

随机B16中，29/40类超过半数batch没有阳性，中位无阳性概率60.53%；直接在小batch上跳过空类并平均会隐式改变病种权重，梯度累积也不会自动生成跨batch pairs或全局AP。本研究使用完整eligible列表解决这个实验/目标估计问题。

若组目标在固定表示上有效，可支持下一阶段检验训练目标/读出的改进；失败只说明这组表征、线性头、预算与正则下未成功，不能证明所有组监督或所有非线性头无效。本阶段不同时修改病史长度、encoder、损失和采样而丢失归因。

## 主要来源

- [RankNet, ICML 2005](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)：pairwise概率排序。
- [ListNet, 2007](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/)：经典listwise框架。
- [SVMap, SIGIR 2007](https://www.cs.cornell.edu/People/tj/publications/yue_etal_07a.pdf)：AP的结构化松弛及AP/AUC区别。
- [Smooth-AP, ECCV 2020](https://www.robots.ox.ac.uk/~vgg/research/smooth-ap/)：可微AP近似。
- [DeepSeekMath, 2024](https://arxiv.org/abs/2402.03300)：原始GRPO的group-relative思想；本研究是面向患者排列的适配。
- [REINFORCE, Williams 1992](https://doi.org/10.1007/bf00992696)：score-function策略梯度。
