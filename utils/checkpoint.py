from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_score: float,
    best_miou: float | None = None,
    raw_model: torch.nn.Module | None = None,
    *,
    checkpoint_type: str | None = None,
    auto_protocol_state: dict | None = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_score": best_score,
        "best_structural_score": best_score,
        "best_mIoU": best_score if best_miou is None else best_miou,
    }
    if raw_model is not None:
        checkpoint["raw_model"] = raw_model.state_dict()
    if checkpoint_type is not None:
        checkpoint["checkpoint_type"] = checkpoint_type
    if auto_protocol_state is not None:
        checkpoint["auto_protocol_state"] = auto_protocol_state
    torch.save(checkpoint, path)


def load_checkpoint_metadata(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Read optional metadata without changing legacy checkpoint loading."""

    checkpoint = torch.load(path, map_location=map_location)
    return {
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "auto_protocol_state": checkpoint.get("auto_protocol_state"),
    }


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
    prefer_raw_model: bool = False,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("raw_model") if prefer_raw_model else None
    if state is None:
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    score = ckpt.get("best_structural_score", ckpt.get("best_score", ckpt.get("best_mIoU", 0.0)))
    return int(ckpt.get("epoch", -1)) + 1, float(score)
