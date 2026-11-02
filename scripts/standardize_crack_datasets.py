"""Create non-destructive DeepCrack-style views for flat crack datasets.

Crack500 and CrackMap store all images/masks in flat directories and assign
samples to splits through text files.  This utility retains those authoritative
sources and creates ``<split>/images`` and ``<split>/masks`` hard-link views.
Each split also receives a provenance manifest preserving the source list order.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root containing Crack500 and CrackMap.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create hard links and manifests. Without this flag the command only validates and reports changes.",
    )
    return parser.parse_args()


def index_by_stem(directory: Path, extensions: set[str]) -> dict[str, Path]:
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions]
    result: dict[str, Path] = {}
    for path in files:
        if path.stem in result:
            raise ValueError(f"Duplicate stem '{path.stem}' in {directory}.")
        result[path.stem] = path
    return result


def read_split_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing split list: {path}")
    stems = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not stems:
        raise ValueError(f"Split list is empty: {path}")
    if len(stems) != len(set(stems)):
        raise ValueError(f"Split list contains duplicate stems: {path}")
    return stems


def manifest_text(dataset: str, split: str, stems: list[str], images: dict[str, Path], masks: dict[str, Path]) -> str:
    lines = ["index\tsource_stem\timage_source\tmask_source"]
    for index, stem in enumerate(stems, start=1):
        lines.append(f"{index:06d}\t{stem}\timages/{images[stem].name}\tmasks/{masks[stem].name}")
    return "\n".join(lines) + "\n"


def link_or_verify(source: Path, destination: Path, execute: bool) -> None:
    if destination.exists():
        if not destination.is_file() or not os.path.samefile(source, destination):
            raise FileExistsError(
                f"Refusing to replace non-matching destination {destination}. Expected hard link to {source}."
            )
        return
    if execute:
        os.link(source, destination)


def write_or_verify_manifest(destination: Path, content: str, execute: bool) -> None:
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite differing manifest: {destination}")
        return
    if execute:
        destination.write_text(content, encoding="utf-8", newline="\n")


def standardize_dataset(root: Path, dataset: str, execute: bool) -> dict[str, int]:
    dataset_root = root / dataset
    image_dir, mask_dir = dataset_root / "images", dataset_root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"{dataset} must contain source directories images/ and masks/.")
    images = index_by_stem(image_dir, IMAGE_EXTENSIONS)
    masks = index_by_stem(mask_dir, MASK_EXTENSIONS)

    split_stems = {split: read_split_list(dataset_root / f"{split}.txt") for split in SPLITS}
    assigned = [stem for stems in split_stems.values() for stem in stems]
    if len(assigned) != len(set(assigned)):
        raise ValueError(f"{dataset} contains a stem assigned to multiple splits.")
    for split, stems in split_stems.items():
        missing_images = [stem for stem in stems if stem not in images]
        missing_masks = [stem for stem in stems if stem not in masks]
        if missing_images or missing_masks:
            raise ValueError(
                f"{dataset}/{split} has missing pairs: images={missing_images[:3]}, masks={missing_masks[:3]}."
            )
    if set(assigned) != set(images) or set(assigned) != set(masks):
        raise ValueError(f"{dataset} source files and split lists do not describe the same stems.")

    counts: dict[str, int] = {}
    for split, stems in split_stems.items():
        split_root = dataset_root / split
        destination_images, destination_masks = split_root / "images", split_root / "masks"
        if execute:
            destination_images.mkdir(parents=True, exist_ok=True)
            destination_masks.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            link_or_verify(images[stem], destination_images / images[stem].name, execute)
            link_or_verify(masks[stem], destination_masks / masks[stem].name, execute)
        write_or_verify_manifest(split_root / "source_order.tsv", manifest_text(dataset, split, stems, images, masks), execute)
        counts[split] = len(stems)
    return counts


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    action = "Standardized" if args.execute else "Validated (dry run)"
    for dataset in ("Crack500", "CrackMap"):
        counts = standardize_dataset(root, dataset, args.execute)
        print(f"{action} {dataset}: " + ", ".join(f"{split}={counts[split]}" for split in SPLITS))


if __name__ == "__main__":
    main()
