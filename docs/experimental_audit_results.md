# 实验审计与效果原因分析

本记录将实验可信度、已经观察到的失败现象、尚未识别的机制分开。当前任务是先建立可信实验，再讨论模型改进；不继续搜索分层窗口、损失权重或集成配方。原提交预测保持不变。

## 固定协议与实验修复

原计划见 [experimental_protocol.md](experimental_protocol.md)，严格确定性修订见 [experimental_protocol_strict.md](experimental_protocol_strict.md)。数据仍为官方2791/365/358患者，训练集独立拟合编码器，输入冻结为BOS+最近511事件，三种子17/42/73，统一预算与mAP选模。固定长病史验证组133人、短组232人，分组不参与选模。

已将初始化权重、CPU/CUDA RNG、每轮患者顺序、实际确定性/AMP/SDP设置、梯度范数、裁剪比例、跳步和eval模式训练指标纳入记录。日志默认可关闭，CPU观察者效应测试确认开启诊断不改变权重、预测或RNG。

三份旧checkpoint重载全部通过：概率最大差小于3e-7，mAP无差异。真实训练患者输入上的CPU适配检查也通过：保留BOS的二维输入到单事件袋适配，其前向、损失、梯度和AdamW单步完全一致；复制有效位置权重后，1024位置容量减到512不改变这些输入的前向。

原warn-only训练复跑中，seed17/42最佳mAP一致；seed73最佳mAP少0.0022125893，超出预先冻结的0.002复现警戒。没有放宽阈值。seed73前7轮训练BCE和验证mAP一致，之后分叉；15轮患者顺序均与独立重建一致，AMP跳步为0。日志明确警告memory-efficient attention backward使用非确定性算法。这证明原设置未强制所有算子确定性，但不足以单独证明此次分叉的全部来源。

按照原协议，后续全部对照统一切换严格确定性条件。旧失败证据保留在 `artifacts/experimental_audit/gate_a_replay_seed73.json`，与新条件结果分开保存。

严格条件下三对完整训练已经全部通过：17/42/73各重复两次，初始化、训练起始RNG、每轮患者顺序、逐轮非时间指标、最佳权重张量以及验证/测试预测均完全一致。证据为 `artifacts/experimental_audit_strict/gate_a_strict_seed{17,42,73}.json`。每对固定a次用于后续对照，b次只检验复现，没有挑分。

严格C0R0三种子的最佳mAP分别为0.21794、0.23450、0.21038，均值0.22094。这是新的数值条件，不能与旧warn-only分数混合估计结构效应。严格设置的目的在于控制实验噪声，而不是提高预测能力；只验证本机当前软件/硬件的重复一致性。

## 旧实验真正说明了什么

以下为已保存的三种子单模型结果重算，来源为 `artifacts/experimental_audit/existing_diagnostics.json`，不使用概率集成。

- 旧raw基线mAP为0.22299±0.01128，新raw为0.20516±0.00962（均值±种子样本标准差）。两者同时改动了attention、读出、BOS/位置约定及参数化/初始化路径，原差值不能单独归因于分层思想。
- 新raw的macro排序下降，但Brier由0.04017改善到0.03857、at-risk BCE由0.17006到0.16149，micro-AP由0.27444到0.28145。概率预测和病种平均排序发生取舍，不能称所有能力全面退化。
- 新raw相对旧raw的净mAP差为−0.017825，其中三个肺癌相关目标贡献−0.010613，约占净下降的59.5%；三者只覆盖10名验证阳性患者，并高度重叠。这是下降位置的分解，不是病因机制证明。
- 三个前列腺相关目标反而贡献+0.007497，共覆盖11人。因此“年龄/性别整体失效”没有得到现有病种结果支持。
- 全体验证40个目标的阳性数中位数为11、最少5；19个目标不超过10人。长史组只有133人，区间更容易变宽。三种子和逐病种报告能暴露脆弱性，不能补出新的患者证据。

## 完整2×2结果

15次严格训练已完成：3对基线复跑（6次）和其余三格各3种子（9次）。所有门禁、源码/数据SHA、共同初始backbone、训练开始RNG、训练配置和患者顺序检查通过。统计见 [factorial_results.json](../artifacts/experimental_audit_strict/factorial_results.json)，逐运行控制见 [audit_summary.json](../artifacts/experimental_audit_strict/audit_summary.json)，逐种子表见 [factorial_scores.csv](../artifacts/experimental_audit_strict/factorial_scores.csv)。

总体验证mAP（每种子单模型后取均值±样本标准差）：

- C0R0，因果attention+原读出：**0.22094±0.01234**。
- C1R0，双向attention+原读出：0.20250±0.00590。
- C0R1，因果attention+mean/last/MLP读出：0.20830±0.00316。
- C1R1，双向attention+mean/last/MLP读出：0.19330±0.00184。

在原读出下改双向，mAP平均少0.01844；在新读出下改双向，平均少0.01500。固定因果attention时换读出，平均少0.01264；固定双向时换读出，平均少0.00920。四个简单效应在17/42/73均为负，不能将C1R1较小的种子标准差解释成效果更好。

跨另一因子两个水平取平均，2000次配对患者bootstrap的总体mAP效应为：

- attention主效应 C：**−0.01672，条件95%区间[−0.03456, −0.00047]**。在当前训练配方/验证患者下有初步退步证据，但上界接近0，且未作多重比较校正，不能外推为双向结构普遍无效。
- 读出包主效应 R：−0.01092，区间[−0.02358, +0.00187]。三种子方向一致，患者不确定区间仍跨0，对新患者的稳定退步尚未识别，更不能把整包效应归因于一个pooling算子。
- mAP交互：+0.00344，区间[−0.02377, +0.02716]。种子交互分别约−0.00673/+0.02605/−0.00901，没有可靠协同证据，也不能认为两项效应精确可加。

新读出比原模型增加33,592个参数，总参数从1,058,544增至1,092,136。两者都包含未参与监督的163,328个language head参数和不可达的位置行；总参数数目不能直接当有效监督容量。新头还改变非线性、聚合、归一化和dropout形状，因此该实验识别的是整包替换在固定训练配方下的效果。

### 概率误差与病种排序确实分歧

在各自mAP选择的checkpoint上，四格Brier分别为0.04016/0.03843/0.03885/0.03899，变体反而更低。C主效应对Brier为−0.000792，区间[−0.001275, −0.000307]；对at-risk BCE为−0.004366，区间[−0.007262, −0.001656]。因此双向替换在这里降低了病种平均AP，却改善了已选checkpoint的概率误差。不能从中直接推出BCE实现错误、类别不平衡是唯一原因，或任意改为focal/ranking loss就会改善。

C主效应的三个肺癌相关目标贡献约−0.00648，约占该净mAP效应的38.8%，仍共用那10名阳性患者；步态异常、脑微出血等目标也有较大贡献。R主效应下降位置不同，主要包含前列腺转移、肥胖等。旧raw与新raw的约60%肺癌贡献和本2×2的38.8%属于不同对比，不能混用。

### 长病史没有得到可靠改善

长组mAP均值C0R0/C1R0/C0R1/C1R1分别为0.21470/0.20411/0.22116/0.19964。C0R1看似比基线高0.00647，但逐种子差为+0.02946/−0.03698/+0.02691，方向不稳。

长组读出主效应仅+0.00099，条件区间[−0.01559,+0.01491]；attention主效应−0.01606，区间[−0.03553,+0.00609]。当前不能声称已改善长史，也不能由长短组分别是否跨0推断长史特异效应。

![三种子单模型mAP及种子离散度](../artifacts/experimental_audit_strict/factorial_metrics.png)

![主效应与交互的配对患者区间](../artifacts/experimental_audit_strict/effects_mAP.png)

## 训练诊断与机制边界

**早停已经生效：每轮按验证mAP保存best.pt，patience=8只决定何时结束训练，最终预测重载最佳轮次。** 因而“后期训练继续拟合、验证变差”只证明存在后期过拟合，不能直接解释已保存最佳模型为什么低分。单凭这些曲线，不足以把过拟合列为当前模型差距的首要原因，也不足以支持先调大正则化。训练/验证差距还可能受样本构成、样本量及指标噪声影响。

原warn-only审计复跑的末轮eval训练mAP为0.8003–0.8639，验证mAP为0.1850–0.2094；训练eval BCE为0.0583–0.0703，而验证BCE为0.1934–0.2021。各自最佳mAP之后，训练拟合继续改善，验证概率误差明显变差。这是过拟合的直接观察，不能只用含dropout、跨更新平均的在线训练BCE替代。

三次复跑均无AMP跳步；最高单轮梯度裁剪比例约1.14%。当前没有发现梯度爆炸或AMP溢出导致训练失败的证据，也没有证据支持立即更换学习率、增加训练轮数或扩大模型。以上并不能证明所有优化设置最优。

早停能够防止使用明显退化的后期权重，但不会保证当前设置下最佳checkpoint已达到最优泛化。mAP选模和BCE选模也不是同一目标；本阶段继续固定mAP规则，不事后改挑BCE最低轮次。若评估正则化，必须在相同早停规则下比较最佳checkpoint的三种子结果，不能拿末轮退化当作收益证据。

独立检查全部12个严格单模型：summary和保存的validation_metrics均等于历史mAP最高值，结束轮次均为最佳轮次+8，没有一次触及35轮上限，最终预测均重载best.pt。可以排除本轮“早停未启用、保存了末轮、预算上限直接截断训练”这些实验错误。早停不保证峰值已足够高，也不能据此排除正则化可能改善峰值。

严格条件下三次基线同样发生过拟合：末轮训练eval mAP为0.7981/0.8987/0.8891，验证mAP为0.1801/0.2139/0.1882；验证BCE从各自mAP最佳轮次的0.1586/0.1747/0.1724上升至0.1946/0.2084/0.2087。现象并未随确定性修复消失。

病史截断是否丢失可预测的远期信息、日期均值压缩是否损失关键事件、全局层收到的局部摘要是否太少，仍是需要独立数学建模与受控对照的问题。观察到截断减少或某组分数变化，不等于识别了这些机制。

## 解读规则

后续最小2×2只改变attention方向与读出包，使用同一输入/BOS、位置容量、共同backbone初始权重及训练起始RNG。读出包增加参数，含mean+last、MLP、归一化及dropout变化，因此只能解释整包效应，不能直接归因到其中某一组件。attention与读出可能交互，不能将串行差值直接当作独立可加贡献。

患者bootstrap区间以已训练的三种子及已选checkpoint为条件，不包含重新训练/未来种子的全部不确定性，也不能消除反复使用验证集的选择偏差。长短组分别报告，并未直接检验长短组的效应差；不能凭“长组显著、短组不显著”声称长史特异改善。区间跨0或种子方向不一致时，保留“尚未识别”的结论。

## BCE/Brier改善而mAP下降：逐病种、逐患者深查

本节主对比固定为 **C1R0−C0R0（只换attention）**，补充 **C0R1−C0R0（只换读出）**；不混用此前factorial主效应的数值。365位验证患者、40目标、三种子各自计算再平均；共14,600个患者×目标cell，其中685阳性、12,588个at-risk阴性、1,327个prior阴性。后者概率固定0且loss差为0。

### 分歧不等于简单“压低所有概率”

主对比mAP 0.220938→0.202502，Brier 0.040161→0.038434，at-risk BCE 0.168581→0.159970。

- ΔBrier=−0.00172715，其中负例项贡献−0.00170254（净改善的98.6%），正例项−0.00002461。负例平均概率在三个种子均上升，而负例二阶矩在seed42/73明显下降，主导了平均收益；seed17的负例Brier反而略差。这符合部分高概率错误尾部被收缩，而非所有负例概率下降。
- ΔBCE=−0.00861075，其中正例项贡献−0.00807708（净改善的93.8%），负例项−0.00053367。正例BCE均值2.37071→2.21420；正例概率几何均值三seed均提高，但算术均值在42/73下降。因为正例logloss是−log(p)，将极小概率抬高会带来很大收益，同时原本高分的正例仍可能下降并失去领先排名。
- 仅作诊断，将每类正/负项各赋1/2权重后，balanced Brier 0.38806→0.39216反而变差，balanced BCE 1.49416→1.42009仍改善。这不是新训练目标或正式选模指标，也不是原发生率下的概率校准指标。
- full Brier本身也等权40病种（每类同365行），不能简单说它只“偏重常见病”。类内大量负例、概率幅值、BCE的对数敏感性，以及AP的头部排序/病种等权共同造成不同变化。

精确正负/病种分解、同患者bootstrap和全部四对contrast见 [score_components.json](../artifacts/ranking_diagnosis/score_components.json) 与 [condition_score_components.csv](../artifacts/ranking_diagnosis/condition_score_components.csv)。在新读出下换attention、或在双向下换读出，并不是Brier/BCE都改善，不能把不同对比混成一个机制。

### 有一部分BCE表象来自不同选模阶段

C0R0的mAP最佳epoch为6/11/11，C1R0为4/6/6。C0R0记录到的最低BCE均值是0.157386，C1R0是0.159411，后者实际上更差。C0R0从BCE最低点走到mAP最佳点的BCE代价为+0.011195，而C1R0仅+0.000559。因此已选权重的BCE差−0.008611可写成：

`(新最低BCE−旧最低BCE) + [(新mAP峰值相对BCE最低的代价)−(旧代价)] = +0.002025−0.010636`。

这是记录轨迹上的描述性恒等式，不是因果分摊。固定epoch4/6/8/10比较时，C1R0−C0R0的BCE差分别为+0.000123/+0.003424/+0.003807/+0.006611，均更差；共同12轮中只有2轮的三种子平均BCE改善。因此不能把已选checkpoint BCE较低解释为双向结构固有概率预测更强。记录没有每epoch验证Brier/患者预测，不能把这个结论自动推广到Brier。

![不同mAP峰值对应不同BCE阶段](../artifacts/ranking_diagnosis/selection_tradeoff.png)

### 哪些病种的排序下降

attention主诊断：

- 肺癌stage1（9阳性）：AP **0.32324→0.15863**。
- 疑似肺癌（10阳性）：AP **0.33577→0.18896**。
- 非小细胞肺癌（9阳性）：AP **0.29765→0.15473**。
- 三个目标合计mAP贡献−0.0113586，占净下降的61.6%，但各自Brier/BCE都改善。三者合计仅10位阳性患者，不是三份独立证据。这里61.6%属于C1R0−C0R0，不是此前C_main的38.8%。
- 脑微出血AP **0.24319→0.11681**，步态异常 **0.21866→0.10877**；两者BCE/Brier也变差，而且阳性集合恰好是同一11人。
- 鼻窦炎、咽炎、支气管炎贡献约48.2%的Brier改善，前列腺部分目标也同时改善AP和概率误差；不是所有疾病一致变差。

读出主诊断不同：因果attention下换读出，前列腺转移（6阳性）AP **0.65022→0.39882**，肥胖（11阳性）**0.30807→0.13949**。两者贡献−0.010499，约占净mAP下降83.1%。双向attention下换读出的主要下降则是前列腺原位癌、前列腺肿瘤。不能把attention的肺癌问题归给读出包。

![病种AP与概率误差变化](../artifacts/ranking_diagnosis/disease_tradeoffs.png)

### 哪些患者、被谁超过

别名Vxxx按原验证患者行序固定，完整UUID保留。以下均为实际保存的预测，不是均值概率集成。

- **V085 = e45345e1-ad83-968c-5eb8-8a42d60ce419**：299事件，全保留。脑微出血在seed42的p从0.61447降至0.06481，rank1→6，上方at-risk阴性0→5，阈值precision 1→1/6。seed17 rank2→8，seed73 rank1→32。阴性V097（df30e63f-374a-277a-83e7-37458e8d5261）三个seed均新超过该患者，尽管V097自身概率也下降，只是V085下降更多。
- **V211 = f0f06352-cf16-1039-9591-e7776eb440c2**：254事件，全保留。NSCLC在seed73的p反而0.23312→0.25380，但rank7→16，上方阴性4→14，阈值precision 3/7→2/16。新增竞争者包括V063（e411e590-96f8-b0d8-b486-5ffed5827293，p0.02446→0.41072）和V237（fae0a9b3-7ec7-486c-93a5-6d603210b70e，p0.11933→0.32473）。该患者正例BCE改善约0.08499，但AP贡献下降。
- **V077 = c9165673-6497-151f-42b9-a33639dd1c36**：52事件，全保留。仅换读出后，肥胖排名三个seed均从1降至23/34/30，上方阴性0→19/30/26；反复超过它的阴性包括V157（89c5ec64-8675-b2ae-d58c-e350f80c722b）。这不是attention对比。
- **V338 = c6de6858-9a3d-7ac5-5d54-0d9888143d69**：123事件，全保留。肺癌stage1在seed73的p从0.000129升至0.027474，正例BCE改善5.361，rank360→69。它显著减少了极端自信的漏报，但仍远未排到列表前列；其余种子不必同向。

V085/V211/V090（cf316fd3-5036-31bf-d8d5-a2db3943c7ea）/V052（a886b328-17b3-0d19-b8ef-96f1462b6c31）的阈值贡献合计占主对比净mAP下降约64.1%。这是有正有负抵消后的净份额，不是绝对损失占比，也不意味着删除四人就会恢复同样分数。

AP被精确写成每个阳性阈值precision的均值；同分时按整个同分组阈值处理。8,220条阳性患者×病种×种子记录重构误差最大2.22e−16。单患者贡献会受其他正例/负例次序影响，是代数分解而非患者干预因果。上述关键下降均同时核对了阴性超越数，避免只看排名。所有正例阈值无精确并列，CSV可float32 round-trip，舍入不是本轮下降原因。

![患者排序变化](../artifacts/ranking_diagnosis/patient_rank_cases.png)

全病例见 [positive_rank_changes.csv](../artifacts/ranking_diagnosis/positive_rank_changes.csv)、[key_cases.json](../artifacts/ranking_diagnosis/key_cases.json)、[patient_loss_changes.csv](../artifacts/ranking_diagnosis/patient_loss_changes.csv)。[negative_overtakes.csv](../artifacts/ranking_diagnosis/negative_overtakes.csv)的attention对比是全量，读出对比仅导出已声明top病种/病例范围，不能把导出行数当四种方法的公平错误总数。

### 概率数值分布与患者次序可以分开诊断

对每个病种只在eligible患者间交换排序和分数分布：保留旧模型患者次序，将新模型从低到高的概率数值依序分配，得到一个仅用于诊断的交叉输出。原AP严格不变，BCE却为**0.159590**、Brier为**0.038383**，相对旧模型均改善，且比实际新输出0.159970/0.038434还稍好。

对两条交换路径对称取平均，BCE总差−0.008611可分为数值分布项−0.009671和患者次序项+0.001060；Brier总差−0.001727分为−0.001936和+0.000209。该代数分解不是唯一的神经机制因果分解，数值分布项也不等于纯校准误差。交叉输出使用固定验证候选的分数集合，不是可部署的校准器，更不能拿它的成绩当新模型成绩。

![概率分布与排序交换诊断](../artifacts/ranking_diagnosis/probability_rank_swap.png)

## 数学问题、经典解法和组信号

记真实条件风险为eta=P(Y=1|X)。Brier的条件期望可写成 `(p−eta)^2+eta(1−eta)`，BCE可写成二元熵加 `KL(Ber(eta)||Ber(p))`；两者总体最优解都是eta。在通常独立二元相关性假设下，真实风险也能提供理想排序。因此不是“BCE天生与AP冲突”，而是有限表征、共享参数、估计与优化误差、不同指标敏感性及选模阶段使结果分离。[Proper scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)、[AP学习的经典讨论](https://www.cs.cornell.edu/People/tj/publications/yue_etal_07a.pdf)。

BCE逐患者惩罚误差；AP评价同病种患者整个列表，头部阳性被阴性挤下会有较大代价。即使理想的独立score梯度都推动正例上升/负例下降，实际参数更新通过共同参数耦合这些score；降低总BCE不保证每个关键正负margin都增大。这是可能的数学机制，不表示本次已测出具体attention内部的Jacobian或表征丢失过程。

经典方法的作用边界：

- **Platt/temperature scaling**处理概率尺度。独立sigmoid下 `p=sigmoid(a*s+b), a>0` 保持同病种患者顺序，不能修复macroAP；非严格单调的isotonic会产生ties，不能混称完全AP不变。后续校准需训练内部独立留出或真正OOF预测，不能用365验证患者拟合。[Guo等, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)。
- **RankNet/RankSVM**显式监督同病种正负患者score margin，适合检查跨患者排序信号；均匀pair错误与AP不同，AUC改善不保证AP改善。[RankNet](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)。
- **ListNet、SVMap、SmoothAP**使用列表或AP松弛/近似，不要求先把训练集展开成独立患者对。把疾病当query、患者当候选、未来新发标签当相关性，可与当前任务对应；这不证明迁移后一定有效。[ListNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/)、[SVMap](https://www.cs.cornell.edu/People/tj/publications/yue_etal_07a.pdf)、[SmoothAP](https://www.robots.ox.ac.uk/~vgg/research/smooth-ap/)。
- **GRPO启发的组信号**也成立，但必须定义正确的随机action。对同一疾病/同一患者列表采多个排列，完整AP作reward，可以用组内相对奖励做policy gradient。仅将组内患者0/1标签减均值，不自动等价于排列AP的策略梯度。原GRPO作用于同prompt多个输出；这里是任务适配，不是原算法复现。[DeepSeekMath](https://arxiv.org/abs/2402.03300)、[REINFORCE](https://doi.org/10.1007/bf00992696)。

当前B16训练中，29/40病种在超过半数随机batch里没有阳性；最稀有病种无阳性概率84.57%。简单加pair/list AP项可能经常没有信号，跳过空类再平均还会改变macro权重。全训练eligible正负pair仅10,725,270，缓存冻结encoder后可做全局或分块列表目标，避免这个问题。数据统计和原始文献见 [classic_method_notes.json](../artifacts/ranking_diagnosis/classic_method_notes.json)。

下一阶段先冻结当前强因果encoder和原linear头的初始化，比较同预算BCE头再优化、RankNet、ListNet适配、SmoothAP、RLOO-AP、GRPO启发AP六组，各三种子。所有新方法共享逐类BCE与ridge规则，主要与BCE再优化对照，避免把额外训练、类权重或正则变化误算成排序收益。具体冻结设置、梯度定标和验收规则见 [group_ranking_protocol.md](group_ranking_protocol.md)。这既测试用户提出的组信号，也保留经典方法参照；不因名称新颖而预判优劣。

## 当前原因判断与后续边界

实验层面已经修复的是确定性、共同输入与初始化、单因素可比性和指标解释；已经核对的是早停/最佳权重、预测对齐、梯度及AMP。没有证据把低分归咎于早停缺失或数值训练崩溃。

当前最具体的退步因素，是在相同早停与训练设置下替换attention方向；证据比“模型太复杂所以过拟合”更直接。但其内部机制、与专门调参的关系仍未识别。读出包存在负向观察，患者区间仍宽，因此不自动继续搜索pooling/MLP直到涨分。

建模层面尚未解决的是：如何在不损失关键事件的情况下利用远期历史、如何选择时间因果归纳偏置，以及如何兼顾病种排序与概率预测。不能用这次对照宣布Hi-BEHRT思想失败，也不能直接从BCE/mAP分歧宣布损失函数错误。后续若研究压缩/分层，应首先保留目前较强的因果attention和原读出作为桥接锚点，只改变一个历史表示因素；需要另立协议，当前未启动。

用户在原因分析之后授权继续检验组内排序信号。后续进入单独冻结的group-ranking研究，不重新搜索病史结构或正则化网格。当前提交预测和此前模型包未替换。

## 可复现入口与记录

```powershell
.venv\Scripts\python.exe scripts/run_strict_audit.py --stage freeze
.venv\Scripts\python.exe scripts/run_strict_audit.py --stage replay
.venv\Scripts\python.exe scripts/run_strict_audit.py --stage factorial
.venv\Scripts\python.exe scripts/analyze_experimental_audit.py --strict --bootstrap 2000
```

已有完整运行只在配置/来源均匹配时复用；部分运行拒绝覆盖。`frozen_manifest.json`包含输入、源码与协议SHA。新实验应明确设置 `deterministic_mode="strict"`；历史配置默认warn-only仅用于保留原复现条件。

排序诊断完成时，代码检查为189项测试、20项subtests通过。审计工具CPU自测曾因硬链接误写一份旧seed17训练起始RNG诊断JSON，已恢复原字段并独立核对；未改权重、预测、配置、输入或严格实验。原整文件SHA未事前记录，不能宣称文件元数据原样恢复；详细依据与限制见 `artifacts/experimental_audit/diagnostic_restore_note.json`。该旧诊断文件不作为新严格归因基线。
