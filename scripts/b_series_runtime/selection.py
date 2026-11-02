from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from pathlib import Path
@dataclass
class UpdateEarlyStopping:
    min_update: int
    patience_updates: int
    min_delta: float
    best_significant_score: float = float("-inf")
    last_significant_update: int = 0

    def update(self, score: float, global_update: int) -> bool:
        if score >= self.best_significant_score + self.min_delta:
            self.best_significant_score = score
            self.last_significant_update = global_update
            return True
        return False

    def should_stop(self, global_update: int) -> bool:
        return global_update >= self.min_update and (global_update - self.last_significant_update) >= self.patience_updates

def is_new_best_checkpoint(score: float, best_checkpoint_score: float) -> bool:
    """Return true only for a strict structural-score improvement.

    Keeping this decision as a small pure helper documents the distinction
    between a checkpoint best and an early-stopping significant best. In
    particular, ties intentionally do not rewrite best_structural_score.pth.
    """

    return score > best_checkpoint_score

def select_auto_protocol_final_checkpoint(
    significant_checkpoint: str | Path,
    absolute_checkpoint: str | Path,
) -> tuple[Path, str]:
    """Choose deployable significant-best, with an explicit-only fallback."""

    significant = Path(significant_checkpoint)
    absolute = Path(absolute_checkpoint)
    if significant.exists():
        return significant, "significant_best"
    if absolute.exists():
        return absolute, "absolute_best_fallback"
    raise FileNotFoundError(
        "Auto Protocol has neither a significant-best nor an absolute-best checkpoint; latest.pth is not a valid final-selection fallback."
    )

@dataclass
class AutoProtocolCheckpointState:
    """Separate diagnostic absolute-best from deployable significant-best."""

    min_delta: float
    absolute_best_score: float = float("-inf")
    absolute_best_update: int | None = None
    significant_best_score: float = float("-inf")
    significant_best_update: int | None = None
    last_significant_update: int = 0

    def consider(self, score: float, global_update: int) -> tuple[bool, bool, float]:
        delta_from_significant = score - self.significant_best_score
        absolute_updated = is_new_best_checkpoint(score, self.absolute_best_score)
        if absolute_updated:
            self.absolute_best_score = score
            self.absolute_best_update = global_update
        significant_updated = score >= self.significant_best_score + self.min_delta
        if significant_updated:
            self.significant_best_score = score
            self.significant_best_update = global_update
            self.last_significant_update = global_update
        return absolute_updated, significant_updated, delta_from_significant

    def state_dict(self) -> dict[str, Any]:
        return {
            "absolute_best_score": self.absolute_best_score,
            "absolute_best_update": self.absolute_best_update,
            "significant_best_score": self.significant_best_score,
            "significant_best_update": self.significant_best_update,
            "last_significant_update": self.last_significant_update,
            "min_delta": self.min_delta,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.absolute_best_score = float(state.get("absolute_best_score", self.absolute_best_score))
        absolute_update = state.get("absolute_best_update")
        self.absolute_best_update = None if absolute_update is None else int(absolute_update)
        self.significant_best_score = float(state.get("significant_best_score", self.significant_best_score))
        significant_update = state.get("significant_best_update")
        self.significant_best_update = None if significant_update is None else int(significant_update)
        self.last_significant_update = int(state.get("last_significant_update", self.last_significant_update))
def structural_score(metrics: dict[str, float], evaluation_cfg: dict) -> float:
    selected = str(evaluation_cfg.get("selection_metric", "structural_score"))
    if selected != "structural_score":
        if selected not in metrics:
            raise KeyError(f"Unknown selection metric: {selected}")
        return metrics[selected]
    weights = evaluation_cfg.get("structural_score_weights", {})
    return (
        float(weights.get("f1", 0.5)) * metrics["F1"]
        + float(weights.get("cldice", 0.3)) * metrics["clDice"]
        + float(weights.get("boundary_f1", 0.2)) * metrics["BoundaryF1"]
    )
