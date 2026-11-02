# TUT Stage4 DPSM80—Transformer 位置诊断

这是受控位置诊断，不是新的最终模块或结构搜索。三行均使用冻结的
DPER+RSDM+LSSCF+DPSM80、Auto Protocol V2 与相同训练协议；仅
`model.stage4_hybrid_order` 和 `save.save_dir` 不同。

| ID | Order | Stage4 张量流 |
|---|---|---|
| P0 | `parallel` | `local(x), routed=DPSM(x), global=Transformer(x)` |
| P1 | `dpsm_then_transformer` | `local(x), routed=DPSM(x), global=Transformer(routed)` |
| P2 | `transformer_then_dpsm` | `local(x), global=Transformer(x), routed=DPSM(global)` |

所有行的 fusion 输入顺序固定为 `[local, routed, global]`，随后均执行
`Refine(x + Fusion(...))`。因此三行参数量、state_dict、初始化、Mamba 核心数
和理论计算图一致；P2 不会交换 fusion 通道语义。三个模式均输出 `route_s4`；
P2 的 route head 输入为 `Transformer(x)`，P0/P1 输入为原始 Stage4 `x`。

协议由服务器上的 TUT 训练集解析，预期为 large：640×640、effective batch 8、
75 epochs、9,225 updates。不得为匹配预期而硬编码。三行训练必须使用
`--skip-final-test`：只按 validation structural score 选择最佳 checkpoint，训练
期间不构造或访问 test split，resolved config 记录 `test_calls: 0`。

如缺少 TUT 结构监督缓存，先执行：

```bash
python scripts/precompute_structural_targets.py --data_root TUT
```

审计命令：

```bash
python tests/test_stage4_hybrid_order.py
python scripts/audit_stage4_hybrid_order.py --device cuda --seed 42 --img_size 640 640
```

三份独立目录的 Auto Protocol dry-run（每次会执行一次 optimizer update，不能在本机误跑）：

```bash
python train.py --config configs/experiments/stage4_order_ablation/tut/P0_Parallel.yaml --device cuda --seed 42 --auto-protocol-dry-run
python train.py --config configs/experiments/stage4_order_ablation/tut/P1_DPSMThenTransformer.yaml --device cuda --seed 42 --auto-protocol-dry-run
python train.py --config configs/experiments/stage4_order_ablation/tut/P2_TransformerThenDPSM.yaml --device cuda --seed 42 --auto-protocol-dry-run
```

随后将三个 dry-run 的 `config.yaml` 传给审计脚本；除 micro batch 与 accumulation 外，
所有冻结 Auto Protocol 字段必须一致，且两者乘积必须仍为 effective batch：

```bash
python scripts/audit_stage4_hybrid_order.py --device cuda --seed 42 --img_size 640 640 --resolved-config P0_CONFIG P1_CONFIG P2_CONFIG
```

筛选规则在运行前冻结：第一轮仅 seed 42，只比较最佳 validation checkpoint 的
structural score，同时报告 F1、mIoU、BoundaryF1、clDice、ComponentError。若
P1/P2 相对 P0 的 structural score 增益不足 0.002，或边界/拓扑指标没有一致改善，
则没有证据改变 parallel；若增益至少 0.002 且 BoundaryF1、clDice 不下降，才补
P0 与胜出方案的多种子。TUT 仅用于筛选；任何正式模型更改必须在 Crack500 的相同
协议下确认。不得因未胜出继续尝试未预注册融合方式，也不得使用 test 选择位置。

正式训练命令：

```bash
python train.py --config configs/experiments/stage4_order_ablation/tut/P0_Parallel.yaml --device cuda --seed 42 --skip-final-test
python train.py --config configs/experiments/stage4_order_ablation/tut/P1_DPSMThenTransformer.yaml --device cuda --seed 42 --skip-final-test
python train.py --config configs/experiments/stage4_order_ablation/tut/P2_TransformerThenDPSM.yaml --device cuda --seed 42 --skip-final-test
```
