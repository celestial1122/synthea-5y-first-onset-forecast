# 分组排序的尺度不变性检验（2026-10-28）

这是在首轮六方法研究完成后、观察到特定数学漏洞而预先定义的单次精炼；不是继续搜索温度、lambda、随机种子直到涨分。首轮原始研究、其源码和全部结果保持冻结。

## 触发证据与假设

`artifacts/group_ranking/gradient_diagnostics.json`只读训练数据证明：将SFT起点logit乘以正数2，所有病种greedy排序与AP严格不变，但固定PL温度下采样macroAP从seed17的0.23056升至0.29425（greedy AP仍0.35011）；seed42从0.45015升至0.59462，seed73从0.44626升至0.57189。

因此采样奖励可以通过更自信而不修复确定性排名来提高。该诊断没有证明首轮实际改进全部来自尺度，亦没有证明去除尺度自由度一定提高验证mAP。新假设是：限制组目标对正仿射logit变换不敏感，可减少这种奖励与部署指标的偏离。

## 唯一目标变化

对每个病种，以全部eligible训练患者计算当前raw logits的均值m和population std sd；缓存SFT初始logits的std sd0并detach。只供排序目标使用：

`z = s + (s - m) * (sd0 / sd - 1)`。

这等于 `m + (s-m)*sd0/sd`。初始化sd=sd0时z=s逐位相同；对任意 `s'=a*s+b, a>0`，z只差公共offset，而PL概率与平滑AP不变。current m/sd必须完整反传，不能detach分母后冒称真正的尺度不变梯度。std≤1e-6、样本不足时显式报错，不静默退回旧目标。

原macroBCE、输出sigmoid和部署排名均使用raw s；只有ranking项看z。原温度仍按初始训练logit std的0.5倍、clamp[0.25,2]固定。组优势标准差归一化（GRPO）与这里的score尺度归一化是两回事。

## 四组×三种子

- bce：同首轮BCE头再优化；作为新代码的匹配控制。
- smoothap_si：只有平滑AP项改看z。
- rloo_ap_si：只有PL组策略改看z，保留raw AP reward与leave-one-out baseline。
- grpo_ap_si：只有PL组策略改看z，保留组内reward标准化。

全部复用首轮同种子17/42/73的SFT因果encoder、133维特征和原linear头初始化；不重训练encoder，不改变511事件输入，不在验证病例上构造奖励或权重。

Adam lr0.003、weight_decay0、ridge mu0.1、初始辅助梯度比例kappa0.25、G8、RL校准4组、calibration seed+27041、training seed+17041、float32 strict、100epoch、patience15、epoch0参与选模、固定epoch10诊断，均与首轮完全相同。lambda按新目标的初始训练梯度重新套用同一规则定标，不根据验证成绩调整。

可复用旧工具函数，但新目标/训练器/runner位于新文件；旧冻结源不修改。新freeze记录首轮manifest、训练尺度诊断、相同cache与新源SHA。新输出根为 `artifacts/group_ranking_scale_invariant`。

## 门禁

1. CPU double证明初始化前向相同、正仿射目标/PL概率不变、eligible score梯度sum=0且与score点积=0、prior零梯度、std退化拒绝；真实CUDA strict forward/backward通过。
2. 新bce三个种子的最佳头、预测、共同逐轮训练/验证指标必须与首轮bce逐位一致。新增reference_std元数据单独核验，不把新增字段误作训练差异。
3. grpo_ap_si-seed17额外完整复跑一次，逐轮非时间指标、头/全模型张量与预测一致；不参与选模。
4. 完成四方法×三种子12次拟合，加1次重复，不按中途结果追加方法。

## 验收和边界

主要比较SI方法对新bce控制的每种子单模型mAP差；补充各SI方法对首轮对应方法的差。共同患者bootstrap2000次，报告总体与固定长/短组、best与epoch10、概率误差、epoch0回退、reward方差、梯度和训练尺度。不得用reward上升代替验证greedy AP。

选择只按三种子平均验证mAP；若需要一个演示checkpoint，种子仍固定42，不挑最高分种子。不得把回退θ0算进步。全部估计条件于已反复使用的验证集，没有消除选模与多方法比较偏差。

若该单次目标精炼仍未超过匹配BCE，则记录未改善并停止本阶段，不继续搜温度/组大小/损失权重。是否解冻encoder、增加训练数据或改变时序结构属于后续独立研究。根目录提交预测和旧模型包继续保留。
