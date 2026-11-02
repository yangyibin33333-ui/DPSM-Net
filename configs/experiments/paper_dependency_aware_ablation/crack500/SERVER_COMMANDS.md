# demamba21：Crack500 strict_v2

正式训练前先通过唯一的配置/构建/前向审计；不要运行 Auto Protocol dry-run：

```bash
python tests/test_paper_dependency_aware_ablation_configs.py
```

全部 PASS 后，在 RTX 4090 D 的真实 `mamba_ssm` 环境按以下顺序执行。每行从
随机初始化开始，使用独立 strict_v2 保存目录；test 不参与选参、种子或检查点。

```bash
python train.py --config configs/experiments/paper_dependency_aware_ablation/crack500/M0_FrameworkOff_DPSMOff_StructSupOff.yaml --seed 42 --device cuda
python train.py --config configs/experiments/paper_dependency_aware_ablation/crack500/M1_FrameworkOn_DPSMOff_StructSupOff.yaml --seed 42 --device cuda
python train.py --config configs/experiments/paper_dependency_aware_ablation/crack500/M2_FrameworkOff_DPSMOn_StructSupOff.yaml --seed 42 --device cuda
python train.py --config configs/experiments/paper_dependency_aware_ablation/crack500/M3_FrameworkOn_DPSMOn_StructSupOff.yaml --seed 42 --device cuda
python train.py --config configs/experiments/paper_dependency_aware_ablation/crack500/M4_FrameworkOff_DPSMOn_StructSupOn.yaml --seed 42 --device cuda
python train.py --config configs/experiments/paper_dependency_aware_ablation/crack500/M5_FrameworkOn_DPSMOn_StructSupOn.yaml --seed 42 --device cuda
python train.py --config configs/experiments/paper_dependency_aware_ablation/crack500/loss_2x2/D1_TangentOn_BPCMOff.yaml --seed 42 --device cuda
python train.py --config configs/experiments/paper_dependency_aware_ablation/crack500/loss_2x2/D2_TangentOff_BPCMOn.yaml --seed 42 --device cuda
```

每次启动核对 resolved config：352×640、effective batch 8、18,816 updates、
初始化 checksum 和独立 save_dir。M0/M1 日志必须显示 `NoDPSMTopoBlock`、
route branch absent 与 `Mamba backend: not applicable`；M2–M5/D1/D2 必须显示
DPSM80 的 4×80 分组和共享 `mamba_ssm ×1`。
