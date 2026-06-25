"""`cvlib cvat-query` — filter/search rows of a CVAT CSV export."""

from __future__ import annotations

import argparse

from loguru import logger

from cv_lib.cli._common import add_verbose

HELP = "Search a CVAT CSV export by task / label / assignee / image."

EPILOG = (
    "Examples:\n"
    "  cvlib cvat-query export.csv --label car\n"
    "  cvlib cvat-query export.csv --task-name batch_3 --assignee anna --count\n"
)

# CLI flag → CSV column name
_FILTERS = {
    "label": "instance_label",
    "task_name": "task_name",
    "assignee": "task_assignee",
    "image": "image_name",
    "job_id": "job_id",
}

_DEFAULT_COLUMNS = ["image_name", "instance_label", "task_name", "task_assignee"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Path to the CVAT CSV export.")
    parser.add_argument("--label", metavar="NAME", help="Filter by instance_label.")
    parser.add_argument("--task-name", metavar="NAME", help="Filter by task_name.")
    parser.add_argument("--assignee", metavar="NAME", help="Filter by task_assignee.")
    parser.add_argument("--image", metavar="NAME", help="Filter by image_name.")
    parser.add_argument("--job-id", metavar="ID", help="Filter by job_id.")
    parser.add_argument(
        "--columns", nargs="+", default=_DEFAULT_COLUMNS, metavar="COL",
        help=f"Columns to print (default: {' '.join(_DEFAULT_COLUMNS)}).",
    )
    parser.add_argument(
        "--count", action="store_true",
        help="Print only the number of matching rows.",
    )
    parser.add_argument(
        "--limit", type=int, default=50, metavar="N",
        help="Max rows to print (default: 50; 0 = no limit).",
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from cv_lib.data.convert import query_cvat_csv

    filters = {
        column: getattr(args, flag)
        for flag, column in _FILTERS.items()
        if getattr(args, flag) is not None
    }

    try:
        rows = query_cvat_csv(args.input, **filters)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    logger.info("Matched {} row(s){}", len(rows), f" for {filters}" if filters else "")

    if args.count:
        print(len(rows))
        return
    if not rows:
        return

    columns = args.columns
    widths = {c: max(len(c), max((len(r.get(c, "")) for r in rows), default=0)) for c in columns}

    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))

    shown = rows if args.limit == 0 else rows[: args.limit]
    for row in shown:
        print("  ".join(row.get(c, "").ljust(widths[c]) for c in columns))

    if len(shown) < len(rows):
        print(f"… and {len(rows) - len(shown)} more (raise --limit to see all)")
