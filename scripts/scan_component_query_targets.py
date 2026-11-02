from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from datasets import scan_train_component_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audit 8-connected DeepCrack component counts for ComponentQuerySwin")
    parser.add_argument("--data_root", default="DeepCrack")
    parser.add_argument("--img_size", type=int, nargs=2, default=(384, 544), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--threshold", type=int, default=127)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = scan_train_component_counts(args.data_root, tuple(args.img_size), args.threshold)
    max_components = max(count for _, count in counts)
    print(f"Train component audit | images: {len(counts)} | max components: {max_components}")
    for stem, count in counts:
        print(f"{stem}: {count}")
    if max_components > 64:
        offenders = [stem for stem, count in counts if count > 64]
        raise ValueError(
            "ComponentQuerySwin supports at most 64 connected components per image; "
            f"offending samples: {', '.join(offenders)}"
        )


if __name__ == "__main__":
    main()
