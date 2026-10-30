# 严格确定性审计修订（2026-10-28）

本修订继承 `experimental_protocol.md` 的固定输入、三种子、选模规则、训练预算、分组、2×2对照与统计方式。原计划和原失败结果保留，不提高复现容差。

## 触发证据

原 warn-only 条件 seed17/42 的最佳 mAP 复现一致；seed73 的最佳 mAP 差为 −0.0022125893，触发预先规定的 0.002 报警。其第1–7轮在线训练BCE与验证mAP完全一致，第8轮首次分叉；最佳轮次从14变为7。

原日志同时含 PyTorch 警告：Memory Efficient attention defaults to a non-deterministic algorithm，并建议 `torch.use_deterministic_algorithms(True, warn_only=False)`。该警告证实旧训练没有强制所有算子确定性，但并不单独证明此次分叉的全部来源。

## 新实验条件和门禁

1. 唯一训练设置变更是 `deterministic_mode="strict"`，调用 `torch.use_deterministic_algorithms(True, warn_only=False)`；继续同一AMP精度、GPU、输入、模型、初始化与训练预算。若算子报错，停止并记录，不自动回退宽松模式。
2. 每个种子17/42/73独立运行两次完整C0R0训练，两次都开启相同诊断。要求初始化权重、训练开始RNG、每轮患者顺序、逐轮非时间训练/验证指标、最佳权重张量指纹及验证预测一致。两次只改变输出目录/名称。
3. 三对重复全部通过后，固定每对的a次作为C0R0，运行其余三格各三种子。b次仅检验复现，不挑高分。所有四格统一严格设置；与旧warn-only成绩仅作背景参考，不作为结构效应估计。
4. 在新目录 `artifacts/experimental_audit_strict` 冻结修订、源码和原数据清单；原失败报告保留。原checkpoint重载和CPU等价性门禁仍适用。

## 结论边界

同环境重复一致只说明这次软硬件和设置下的可重复性，不保证跨GPU、PyTorch版本或平台逐位一致。确定性会消除部分实验噪声，不被当作提高真实泛化能力的方法。最终仍以三种子单模型结果和共同患者bootstrap区间报告效应；区间宽或方向不一致时不强行归因。
