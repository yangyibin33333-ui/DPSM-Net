# DPSM-Net

Code accompanying the paper *Efficient Infrastructure Crack Segmentation with Detail-Preserving Dual-Resolution Encoding and Direction-Partitioned Shared State-Space Modeling*.

## Datasets

This repository does not redistribute datasets. Obtain them from the original releases and comply with their respective terms:

- [DeepCrack](https://github.com/yhlleo/DeepCrack)
- [CamCrack789](https://github.com/lovelyyoshino/RHACrackNet)
- [CrackMap](https://github.com/ikatsamenis/CrackMap)
- [Crack500](https://github.com/fyangneil/pavement-crack-detection)
- [CrackForest (CFD)](https://github.com/cuilimeng/CrackForest-dataset)
- [TUT](https://github.com/Karl1109/TUT)

## Splits

The splits/ files list the image filename stems used in the paper, sorted by dataset and split. TUT uses the partition released by its authors; the other five datasets were re-partitioned. Organize images and matching masks as `<dataset>/<split>/images/` and `<dataset>/<split>/masks/`, with matching stems. The manifests record the intended membership; the dataset loader reads the corresponding directories.

## Derived caches

The tangent and other structural target caches in `<split>/targets/` are derived from masks and are not included. Generate them with:

~~~bash
python scripts/precompute_structural_targets.py --data_root /path/to/DeepCrack
~~~

## Installation

~~~bash
pip install -r requirements.txt
~~~

The formal Mamba implementation requires a compatible Linux, CUDA, and PyTorch environment; follow the mamba-ssm project's installation instructions.

## Reproduction

The frozen DPSM80 protocol is provided in configs/experiments/auto_protocol/C40_DPSM80_AUTO_PROTOCOL_V2.yaml. Replace the dataset and checkpoint paths for your environment:

~~~bash
python train.py --config configs/experiments/auto_protocol/C40_DPSM80_AUTO_PROTOCOL_V2.yaml --data_root /path/to/DeepCrack --save_dir ./runs/dpsm80 --seed 42 --device cuda
python test.py --config configs/experiments/auto_protocol/C40_DPSM80_AUTO_PROTOCOL_V2.yaml --data_root /path/to/DeepCrack --checkpoint ./runs/dpsm80/best_structural_score.pth --split test --device cuda
~~~

Checkpoints are not included. The example checkpoint path should match the file selected by your training run. scripts/bootstrap_b_series_20261023.py is an optional historical study helper; set B_SERIES_EXTERNAL_ROOT to the external source directory if it is needed.

## Attribution

Mamba and attention components refer to the original public implementations and papers cited in the paper. Third-party dependencies and datasets retain their own licenses.
