from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import DEMambaNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Profile Lite-TopoRoute-Mamba parameters and batch-1 latency")
    parser.add_argument("--config", default="configs/demambanet.yaml")
    parser.add_argument("--img_size", type=int, nargs=2, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)
    model = DEMambaNet(**model_cfg).to(device).eval()
    height, width = args.img_size or cfg["data"]["img_size"]
    image = torch.randn(1, 3, height, width, device=device)
    with torch.no_grad():
        for _ in range(args.warmup):
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        for _ in range(args.iterations):
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    latency_ms = (perf_counter() - start) * 1000.0 / args.iterations
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"Batch-1 latency ({height}x{width}, {device}): {latency_ms:.2f} ms")
    for stage, backends in model.mamba_backend_info().items():
        print(f"Mamba backend {stage}: {', '.join(backends)}")


if __name__ == "__main__":
    main()
