"""`cvlib dedup` — find near-duplicate images and cross-split data leakage."""

from __future__ import annotations

import argparse

from cv_lib.cli._common import add_verbose

HELP = "Find near-duplicate images (pHash) or train/val/test data leakage."

EPILOG = (
    "Examples:\n"
    "  cvlib dedup images/                 # duplicate clusters within a folder\n"
    "  cvlib dedup dataset/ --leakage      # cross-split dups in dataset/images/<split>\n"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="Images directory, or dataset root with --leakage.")
    parser.add_argument(
        "--leakage", action="store_true",
        help="Check images/<split> folders for cross-split near-duplicates.",
    )
    parser.add_argument(
        "--hamming", type=int, default=5,
        help="Max Hamming distance to treat hashes as duplicates (default: 5).",
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> int:
    from cv_lib.data.dedup import check_split_leakage, find_duplicates

    if args.leakage:
        pairs = check_split_leakage(args.path, hamming_threshold=args.hamming)
        print(f"Split leakage: {len(pairs)} cross-split near-duplicate pair(s)")
        for p in pairs:
            print(f"  [{p.split_a}] {p.path_a.name}  ==  [{p.split_b}] {p.path_b.name}  (d={p.distance})")
        return 1 if pairs else 0

    clusters = find_duplicates(args.path, hamming_threshold=args.hamming)
    total = sum(len(c) for c in clusters)
    print(f"Duplicates: {len(clusters)} cluster(s), {total} image(s)")
    for i, cluster in enumerate(clusters, 1):
        print(f"  cluster {i}:")
        for path in cluster:
            print(f"    {path}")
    return 1 if clusters else 0
