# Crack500 依赖感知主消融（strict_v2，M0–M5）

`strict_v2` 是当前论文主消融归档。它用严格拓扑匹配的
`NoDPSMTopoBlock` 替代旧 M0/M1 的 `LocalCrackBlock`：保留 Stage4 的
`DWConvBlock + StripConvBlock` local branch、`TransformerGlobalBlock` global
branch、residual 和 refine；完全删除 route Mamba、route head 与 route logits；
仅以 `local + transformer` 的 640→320 两分支融合输出。它不是 Identity，也
不保留伪造的第三个零分支。

| ID | Framework | DPSM80 | StructSup | Stage4 | route_s4 |
|---|---:|---:|---:|---|---|
| M0 | × | × | × | NoDPSMTopoBlock | 无 |
| M1 | ✓ | × | × | NoDPSMTopoBlock | 无 |
| M2 | × | ✓ | × | TopoRouteBlock + DPSM80 | 有 |
| M3 / D0 | ✓ | ✓ | × | TopoRouteBlock + DPSM80 | 有 |
| M4 | × | ✓ | ✓ | TopoRouteBlock + DPSM80 | 有 |
| M5 / D3 | ✓ | ✓ | ✓ | TopoRouteBlock + DPSM80 | 有 |

`Framework` 开启时固定为 LSSCF+DPER+RSDM；RSDM 仍只处理 DPER 的
半分辨率 40-channel semantic stream。`StructSup` 仅可与 DPSM80 同时开启：
`lambda_tangent=0.2`、`lambda_bpcm=0.025`。D1 只开启 tangent，D2 只开启
BPCM，二者都以 M3 为基础。M0/M1 的 tangent 和 BPCM 必须为零，且训练前会
拒绝非法的严格 NoDPSM 结构监督配置。

新旧结果不得混用：旧
`runs/paper/dependency_aware_ablation/.../M0|M1...` 是 LocalCrackBlock
Stage4 替换实验，必须保留，但不能填入 strict_v2 主表。八个 strict_v2 配置
均使用独立 `runs/paper/dependency_aware_ablation_strict_v2/crack500/...`。

所有配置共享 Crack500 Frozen Auto Protocol V2：原生解析输入 352×640、
effective batch 8、target updates 18,816、64 epochs、SCNP warmup 941 updates、
BPCM start/ramp 5,645/3,763 updates、每 400 updates 验证、minimum early update
10,349、patience 2,823、`min_delta=0.0005`。数据划分、增强、优化器、EMA、
验证、早停、检查点规则不变；每次训练随机初始化，test 不参与选择。

M5 的行为仍与冻结 A5 相同；测试会排除仅为显式默认值的
`use_topology_matched_no_dpsm_stage4: false` 和 `save_dir` 后，核验模型
state_dict 与固定输入输出完全一致。
