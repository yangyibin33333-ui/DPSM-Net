# OBSOLETE — Crack500 G0–G7 三因素注册表

> 此目录保留为不可删除的历史注册资料，不再是正式实验入口。它错误地将带四个全宽方向 Mamba 的 `TopoRouteBlock` 作为“DPSM off”，并允许在无 DPSM 的情况下把 tangent+BPCM 记为 StructSup；实际 tangent 会因无有效的普通 Stage4 route 语义而失效。请使用 `configs/experiments/paper_dependency_aware_ablation/crack500/` 的 M0–M5。

本目录冻结论文主表的 `Framework × DPSM × StructSup` 完整 `2^3` 组合。所有行均从随机初始化开始（已有 G3/G7 仅登记为可复用的同协议 seed-42 结果），并继承 Crack500 Auto Protocol V2：352×640、effective batch 8、64 epochs、18,816 updates、SCNP warmup 941 updates、BPCM start/ramp 5,645/3,763 updates、每 400 updates 验证、minimum early update 10,349、patience 2,823、`min_delta=0.0005`。训练命令固定使用 `--seed 42`；test split 不参与参数、种子或检查点选择。

| ID | Framework | DPSM | StructSup | 配置/旧结果映射 |
|---|---:|---:|---:|---|
| G0 | × | × | × | 新增，A1 架构，两项结构监督为 0 |
| G1 | ✓ | × | × | 新增，A4 架构，两项结构监督为 0 |
| G2 | × | ✓ | × | 新增，A1 架构 + DPSM80，两项结构监督为 0 |
| G3 | × | × | ✓ | 与旧 A1 等价；可复用旧 seed-42 结果 |
| G4 | ✓ | ✓ | × | 新增；同时是损失内部 2×2 的 D0 |
| G5 | ✓ | × | ✓ | 与旧 A4 等价的独立复跑入口；旧 NaN 记录保留且不覆盖 |
| G6 | × | ✓ | ✓ | 新增，A1 架构 + DPSM80 |
| G7 | ✓ | ✓ | ✓ | 与旧 A5 等价；可复用旧 seed-42 结果，同时是 D3 |

## 因素定义和真实依赖

- **Framework off** 是 A1：普通 decoder，LSSCF/DPER/RSDM 均关闭。**Framework on** 是 A4 的完整 LSSCF+DPER+RSDM。RSDM 只处理 DPER 的半分辨率 40 通道语义流，不能移植到普通 decoder 全分辨率 Stage1；A3→A4 仍是 RSDM 的内部条件证据。
- **DPSM off** 保留普通 Stage4 `TopoRouteBlock` 的四个独立全宽方向 Mamba 和 `route_s4`。**DPSM on** 只在相同的 320 通道 Stage4 中改为四个 80 通道方向组、batch packing 和一个共享 Mamba 核心；外层 local/global 分支、route head、融合不变。实现仅解除旧的 Framework 注册保护，不改变 DPSM 数学路径。
- **StructSup off** 仅设 `lambda_tangent=0`、`lambda_bpcm=0`；**on** 为冻结的 `0.2`、`0.025`。普通 Stage4 和 DPSM 都输出 `route_s4`，所以 tangent 不依赖 DPSM；mask、edge、SCNP、增强、优化器、EMA、更新时钟、验证和检查点规则不变。

`loss_2x2/D1_TangentOn_BPCMOff.yaml` 和 `loss_2x2/D2_TangentOff_BPCMOn.yaml` 补齐最终 Framework+DPSM 架构的内部损失 2×2；D0/G4、D3/G7 直接映射，不复制配置。

各 `save_dir` 均是 `runs/paper/three_factor_ablation/crack500/...` 下的唯一目录。G5 的可选 `--non-finite-diagnostic` 是 CLI 只读诊断，不写入 YAML，默认关闭，因此正常数值路径、loss、梯度、optimizer、scheduler 和 EMA 均不变。首次非有限事件会记录 epoch/batch/update、loss 分解、mask/edge/route_s4 有限性、scaler 前后值、实际 step、学习率及 SCNP/BPCM 权重；之后仅累计日志。

运行前请执行 `python tests/test_paper_three_factor_ablation_configs.py`。该审计拒绝任何非注册字段变化，并检查 G3/G5/G7 对旧 A1/A4/A5 的等价性。fvcore FLOPs 是下界；所有正式训练和效率测量在 demamba21 的 RTX 4090 D、真实 `mamba_ssm` 环境执行。
