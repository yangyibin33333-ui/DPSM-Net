"""Create non-destructive DeepCrack-style dataset views for CFD and TUT.

TUT already has official train/validation/test directories.  This utility
creates hard-link views under ``<split>/images`` and ``<split>/masks``.

CFD (CrackForest) supplies 118 labelled image/.mat/.seg triples plus 37
unlabelled images.  The .seg annotations are parsed into 0/255 PNG masks and
the labelled subset is split deterministically (seed 42) in a foreground-rate
stratified 83/12/23 train/validation/test partition.  Raw files are never
moved or deleted; unlabelled CFD images stay in ``CFD/image`` and are recorded
in ``CFD/unlabeled_images.tsv`` instead of being silently included.
"""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CFD_SPLIT_COUNTS = {"train": 83, "val": 12, "test": 23}
CFD_SPLIT_SEED = 42


@dataclass(frozen=True)
class Pair:
    stem: str
    image: Path
    mask_source: Path
    foreground_fraction: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root containing CFD and TUT.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create hard links, converted CFD masks, and manifests. Without this flag the command only audits sources.",
    )
    return parser.parse_args()


def index_by_stem(directory: Path, extensions: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    result: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate stem '{path.stem}' in {directory}.")
        result[path.stem] = path
    return result


def link_or_verify(source: Path, destination: Path, execute: bool) -> None:
    if destination.exists():
        if not destination.is_file() or not os.path.samefile(source, destination):
            raise FileExistsError(f"Refusing to replace non-matching destination {destination}. Expected hard link to {source}.")
        return
    if execute:
        os.link(source, destination)


def write_or_verify_text(destination: Path, content: str, execute: bool) -> None:
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite differing manifest: {destination}")
        return
    if execute:
        destination.write_text(content, encoding="utf-8", newline="\n")


def manifest_text(pairs: Iterable[Pair], dataset_root: Path, split: str) -> str:
    lines = ["index\tsource_stem\timage_source\tmask_source\tforeground_fraction"]
    for index, pair in enumerate(sorted(pairs, key=lambda item: item.stem), start=1):
        image_source = pair.image.relative_to(dataset_root).as_posix()
        mask_source = pair.mask_source.relative_to(dataset_root).as_posix()
        fraction = "" if pair.foreground_fraction is None else f"{pair.foreground_fraction:.8f}"
        lines.append(f"{index:06d}\t{pair.stem}\t{image_source}\t{mask_source}\t{fraction}")
    return "\n".join(lines) + "\n"


def create_standard_view(dataset_root: Path, split: str, pairs: Iterable[Pair], execute: bool) -> None:
    pairs = tuple(sorted(pairs, key=lambda item: item.stem))
    image_destination = dataset_root / split / "images"
    mask_destination = dataset_root / split / "masks"
    if execute:
        image_destination.mkdir(parents=True, exist_ok=True)
        mask_destination.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        link_or_verify(pair.image, image_destination / pair.image.name, execute)
        link_or_verify(pair.mask_source, mask_destination / pair.mask_source.name, execute)
    write_or_verify_text(dataset_root / split / "source_order.tsv", manifest_text(pairs, dataset_root, split), execute)


def standardize_tut(root: Path, execute: bool) -> dict[str, int]:
    dataset_root = root / "TUT"
    counts: dict[str, int] = {}
    seen_stems: set[str] = set()
    for split in SPLITS:
        images = index_by_stem(dataset_root / f"{split}_img", IMAGE_EXTENSIONS)
        masks = index_by_stem(dataset_root / f"{split}_lab", IMAGE_EXTENSIONS)
        missing_masks, missing_images = set(images) - set(masks), set(masks) - set(images)
        if missing_masks or missing_images:
            raise ValueError(f"TUT/{split} has unpaired files: missing_masks={sorted(missing_masks)[:3]}, missing_images={sorted(missing_images)[:3]}.")
        overlap = seen_stems & set(images)
        if overlap:
            raise ValueError(f"TUT contains stems assigned to multiple splits: {sorted(overlap)[:3]}.")
        pairs: list[Pair] = []
        for stem in sorted(images):
            if Image.open(images[stem]).size != Image.open(masks[stem]).size:
                raise ValueError(f"TUT/{split}/{stem} image and mask dimensions differ.")
            pairs.append(Pair(stem, images[stem], masks[stem]))
        create_standard_view(dataset_root, split, pairs, execute)
        counts[split] = len(pairs)
        seen_stems.update(images)
    return counts


def parse_cfd_seg(path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    lines = path.read_text(encoding="ascii").splitlines()
    try:
        data_index = lines.index("data")
    except ValueError as error:
        raise ValueError(f"{path} lacks a data section.") from error
    metadata: dict[str, str] = {}
    for line in lines[:data_index]:
        key, value = line.split(" ", 1)
        metadata[key] = value
    try:
        width, height, segment_count = int(metadata["width"]), int(metadata["height"]), int(metadata["segments"])
    except KeyError as error:
        raise ValueError(f"{path} lacks required metadata: {error.args[0]}.") from error
    if image_size != (width, height):
        raise ValueError(f"{path} is {width}x{height}, but its image is {image_size[0]}x{image_size[1]}.")

    coverage = np.zeros((height, width), dtype=bool)
    foreground = np.zeros((height, width), dtype=bool)
    for line in lines[data_index + 1 :]:
        segment, row, start, end = map(int, line.split())
        if not (0 <= segment < segment_count and 0 <= row < height and 0 <= start <= end < width):
            raise ValueError(f"{path} contains an invalid segment run: {line!r}.")
        if coverage[row, start : end + 1].any():
            raise ValueError(f"{path} contains overlapping segment runs.")
        coverage[row, start : end + 1] = True
        if segment > 0:
            foreground[row, start : end + 1] = True
    if not coverage.all():
        raise ValueError(f"{path} does not cover every image pixel.")
    return foreground.astype(np.uint8) * 255, float(foreground.mean())


def cfd_stratified_split(pairs: list[Pair], seed: int) -> dict[str, list[Pair]]:
    if len(pairs) != sum(CFD_SPLIT_COUNTS.values()):
        raise ValueError(f"Expected {sum(CFD_SPLIT_COUNTS.values())} paired CFD samples, received {len(pairs)}.")
    ordered = sorted(pairs, key=lambda item: (item.foreground_fraction, item.stem))
    rng = random.Random(seed)
    remaining = CFD_SPLIT_COUNTS.copy()
    assigned = {split: [] for split in SPLITS}
    for index, pair in enumerate(ordered, start=1):
        slots_left = len(ordered) - index + 1
        forced = [split for split in SPLITS if remaining[split] == slots_left]
        candidates = forced or [split for split in SPLITS if remaining[split] > 0]
        # The largest deficit from the ideal cumulative split ratio keeps each
        # split distributed across low- and high-foreground samples. Seeded
        # random tie-breaking makes the result deterministic and reproducible.
        scores = {
            split: CFD_SPLIT_COUNTS[split] * index / len(ordered) - len(assigned[split])
            for split in candidates
        }
        best = max(scores.values())
        tied = [split for split in candidates if abs(scores[split] - best) < 1e-12]
        split = rng.choice(tied)
        assigned[split].append(pair)
        remaining[split] -= 1
    if remaining != {split: 0 for split in SPLITS}:
        raise RuntimeError(f"CFD split allocation failed: {remaining}")
    return assigned


def write_or_verify_cfd_mask(path: Path, expected: np.ndarray, execute: bool) -> None:
    if path.exists():
        actual = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise FileExistsError(f"Refusing to replace non-matching generated mask {path}.")
        return
    if execute:
        Image.fromarray(expected, mode="L").save(path)


def standardize_cfd(root: Path, execute: bool) -> dict[str, int]:
    dataset_root = root / "CFD"
    images = index_by_stem(dataset_root / "image", IMAGE_EXTENSIONS)
    matrices = index_by_stem(dataset_root / "groundTruth", {".mat"})
    segments = index_by_stem(dataset_root / "seg", {".seg"})
    if set(matrices) != set(segments):
        raise ValueError("CFD .mat and .seg annotation stems differ.")
    paired_stems = sorted(set(images) & set(segments))
    unmatched_images = sorted(set(images) - set(segments))
    unexpected_annotations = sorted(set(segments) - set(images))
    if unexpected_annotations:
        raise ValueError(f"CFD has annotations without images: {unexpected_annotations[:3]}.")

    cache_directory = dataset_root / "derived_masks_from_seg"
    if execute:
        cache_directory.mkdir(parents=True, exist_ok=True)
    pairs: list[Pair] = []
    for stem in paired_stems:
        mask, fraction = parse_cfd_seg(segments[stem], Image.open(images[stem]).size)
        cache_path = cache_directory / f"{stem}.png"
        write_or_verify_cfd_mask(cache_path, mask, execute)
        pairs.append(Pair(stem, images[stem], cache_path, fraction))

    assigned = cfd_stratified_split(pairs, CFD_SPLIT_SEED)
    for split in SPLITS:
        create_standard_view(dataset_root, split, assigned[split], execute)
    unlabelled_text = "source_stem\timage_source\n" + "".join(
        f"{stem}\timage/{images[stem].name}\n" for stem in unmatched_images
    )
    write_or_verify_text(dataset_root / "unlabeled_images.tsv", unlabelled_text, execute)
    return {split: len(assigned[split]) for split in SPLITS}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    action = "Standardized" if args.execute else "Validated (dry run)"
    tut = standardize_tut(root, args.execute)
    cfd = standardize_cfd(root, args.execute)
    print(f"{action} TUT: " + ", ".join(f"{split}={tut[split]}" for split in SPLITS))
    print(f"{action} CFD: " + ", ".join(f"{split}={cfd[split]}" for split in SPLITS) + "; unlabeled_raw_images=37")


if __name__ == "__main__":
    main()
