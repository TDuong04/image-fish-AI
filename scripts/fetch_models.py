#!/usr/bin/env python
"""Download the trained checkpoints from the GitHub release, verified.

    python scripts/fetch_models.py              # all six
    python scripts/fetch_models.py --only 960   # just the 960px models
    python scripts/fetch_models.py --check      # verify what is already here

Weights are not in git -- 37 MB of binaries would be carried by every clone
forever, and they are release artifacts rather than source. They live on the
release instead, and this fetches them.

Every file is checked against the SHA-256 in docs/model_manifest.json. A
truncated download is worse than a missing one: it loads, runs, and quietly
produces wrong detections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fish import paths  # noqa: E402

RELEASE = "v0.1.0-models"
BASE = f"https://github.com/TDuong04/image-fish-AI/releases/download/{RELEASE}"
MANIFEST = paths.DOCS_DIR / "model_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        raise SystemExit(f"Missing {paths.rel(MANIFEST)} -- it ships with the repo.")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r    {done/1e6:5.1f}/{total/1e6:.1f} MB ({pct:3.0f}%)",
                          end="", flush=True)
            print()
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"\n{url}\nHTTP {exc.code}. If the release moved, update RELEASE in this file."
        ) from None
    # Rename only after the whole body arrived, so a killed download leaves no
    # file that looks complete.
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("640", "960"), help="fetch one resolution only")
    ap.add_argument("--check", action="store_true", help="verify local files, download nothing")
    ap.add_argument("--force", action="store_true", help="re-download even if present and valid")
    args = ap.parse_args()

    entries = load_manifest()
    if args.only:
        entries = [e for e in entries if f"-{args.only}-" in e["file"]]

    paths.MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    ok = missing = fixed = bad = 0

    for e in entries:
        dest = paths.MODELS_ROOT / e["file"]
        label = f"{e['file']}  (mAP50 {e['map50']})"

        if dest.exists() and not args.force:
            if sha256(dest) == e["sha256"]:
                print(f"  [ ok ] {label}")
                ok += 1
                continue
            print(f"  [BAD ] {label} -- checksum mismatch")
            if args.check:
                bad += 1
                continue
            print("         re-downloading")
        elif args.check:
            print(f"  [miss] {label}")
            missing += 1
            continue

        print(f"  [get ] {label}")
        download(f"{BASE}/{e['file']}", dest)
        if sha256(dest) != e["sha256"]:
            raise SystemExit(f"\n{e['file']} downloaded but the checksum does not match.\n"
                             f"Delete it and retry; do not use it.")
        fixed += 1

    print()
    if args.check:
        print(f"{ok} verified, {missing} missing, {bad} corrupt"
              + ("" if not (missing or bad) else "  -- run without --check to fix"))
        return 1 if (missing or bad) else 0

    print(f"{ok + fixed} checkpoint(s) in {paths.rel(paths.MODELS_ROOT)}/ "
          f"({fixed} downloaded, {ok} already present)")
    print("\nUse each at the resolution it was trained at -- 640 models at 640, 960 at 960.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
