#!/usr/bin/env python
"""Archive files we no longer use, safely: move, never delete, and keep a manifest that can undo it.

    python scripts/archive_paths.py data/interim/ozfish_labelled --reason "duplicate of data/processed/ozfish"   # dry run
    python scripts/archive_paths.py data/interim/ozfish_labelled --reason "..." --yes                            # do it
    python scripts/archive_paths.py --restore data/archive/20261006-0900/MANIFEST.json                           # undo

Built for a shared machine where nothing may be lost or surprised. Everything moved goes into
`<DATA_ROOT>/archive/<timestamp>/` (a rename on the same disk, so it is instant and uses no extra space), and
`MANIFEST.json` there records each original path, size and reason. Nothing is deleted or compressed.
It refuses the datasets, raw archives and interim sources the project depends on (PROTECTED below), refuses
paths outside DATA_ROOT unless --allow-outside is given, and does nothing without --yes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fish import paths  # noqa: E402

PROTECTED = [
    "processed/deepfish", "processed/kakadu", "processed/ozfish_train", "processed/ozfish_eval", "processed/smoke",
    "raw/deepfish_darknet.zip", "raw/ozfish_test.zip", "raw/kakadu_training_data.zip", "raw/labelled-zip.zip",
    "interim/deepfish_darknet", "interim/ozfish_test", "archive",
]


def size_of(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def protected(p: Path) -> str | None:
    root = paths.DATA_ROOT.resolve()
    try:
        rel = p.resolve().relative_to(root).as_posix()
    except ValueError:
        return None
    for prot in PROTECTED:
        if rel == prot or rel.startswith(prot + "/") or prot.startswith(rel + "/"):
            return prot
    return None


def restore(manifest: Path, yes: bool) -> int:
    data = json.loads(manifest.read_text())
    for item in data["items"]:
        src, dst = Path(item["archived"]), Path(item["original"])
        if not src.exists():
            print(f"missing in archive, skipped: {src}")
            continue
        if dst.exists():
            sys.exit(f"cannot restore {dst}: something already exists there. Nothing was changed for it.")
        print(f"{'restoring' if yes else 'would restore'}: {src} -> {dst}")
        if yes:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    if not yes:
        print("dry run: nothing moved. Add --yes to restore.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--reason", default=None, help="why these are being archived (recorded in the manifest)")
    ap.add_argument("--yes", action="store_true", help="actually move; without it only a plan is printed")
    ap.add_argument("--allow-outside", action="store_true", help="allow paths outside DATA_ROOT (still never deleted)")
    ap.add_argument("--restore", type=Path, default=None, help="undo a previous archive from its MANIFEST.json")
    args = ap.parse_args()

    if args.restore:
        return restore(args.restore, args.yes)
    if not args.paths or not args.reason:
        sys.exit("give at least one path and a --reason (or --restore MANIFEST.json).")

    root = paths.DATA_ROOT.resolve()
    plan = []
    for raw in args.paths:
        p = raw if raw.is_absolute() else Path.cwd() / raw
        if not p.exists():
            sys.exit(f"does not exist: {raw}")
        hit = protected(p)
        if hit:
            sys.exit(f"refusing {raw}: it is, or contains, the protected path 'data/{hit}'.")
        inside = True
        try:
            p.resolve().relative_to(root)
        except ValueError:
            inside = False
        if not inside and not args.allow_outside:
            sys.exit(f"refusing {raw}: outside DATA_ROOT ({root}). Use --allow-outside if you are sure.")
        plan.append(p)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = root / "archive" / stamp
    n = 1
    while dest.exists():                # two runs in the same second must never share a folder or overwrite a manifest
        n += 1
        dest = root / "archive" / f"{stamp}-{n}"
    items, total = [], 0
    for p in plan:
        n = size_of(p)
        total += n
        target = dest / (p.resolve().relative_to(root).as_posix() if p.resolve().is_relative_to(root) else "outside/" + p.name)
        items.append({"original": str(p.resolve()), "archived": str(target), "bytes": n, "is_dir": p.is_dir()})
        print(f"{'moving' if args.yes else 'would move'}: {p}  ({n / 1e6:.1f} MB)\n    -> {target}")
    print(f"total {total / 1e9:.2f} GB; reason: {args.reason}")
    if not args.yes:
        print("dry run: nothing moved. Add --yes to do it.")
        return 0

    for it in items:
        target = Path(it["archived"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(it["original"], str(target))
    (dest / "MANIFEST.json").write_text(json.dumps({"created": stamp, "reason": args.reason, "items": items}, indent=2))
    print(f"done. Undo with: python scripts/archive_paths.py --restore {paths.rel(dest / 'MANIFEST.json')} --yes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
