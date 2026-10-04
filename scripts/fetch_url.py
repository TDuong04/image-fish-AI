#!/usr/bin/env python
"""Download one large file over several connections, resumably, and verify its checksum.

    python scripts/fetch_url.py URL OUT --md5 c282e3f0... --connections 8

Some hosts (Zenodo, for one) give each connection only a few hundred KB/s, so a single
`curl` of a multi-GB archive takes many hours. This splits the file into 32 MB chunks, fetches
them with HTTP Range requests on several connections, and records finished chunks in
`OUT.parts` so an interrupted run resumes instead of restarting. Refuses to start without
enough free disk, and refuses a server that ignores Range. Standard library only, so it runs
unchanged on the workstation, the Linux GPU server and the Jetson.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHUNK = 32 * 1024 * 1024   # default; --chunk-mb overrides


def probe(url: str) -> int:
    """Total size, and proof that the server honours Range."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.status != 206:
            sys.exit(f"server answered {r.status} to a Range request; it does not support resuming or "
                     "parallel fetches. Use curl instead.")
        return int(r.headers["Content-Range"].split("/")[1])


def fetch_chunk(url: str, out: Path, index: int, total: int, lock: threading.Lock, done: set,
                parts: Path, counter: list, chunk: int) -> None:
    start = index * chunk
    end = min(start + chunk, total) - 1
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read: {len(data)} of {end - start + 1}")
            with lock:
                with open(out, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                done.add(index)
                parts.write_text(json.dumps(sorted(done)))
                counter[0] += len(data)
            return
        except Exception as exc:  # network errors are routine on long downloads
            time.sleep(2 * (attempt + 1))
            last = exc
    raise SystemExit(f"chunk {index} failed after 6 attempts: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("out", type=Path)
    ap.add_argument("--md5", default=None, help="expected MD5; the file is verified at the end")
    ap.add_argument("--connections", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=32, help="smaller chunks show progress sooner and lose less on a failure")
    args = ap.parse_args()

    chunk = args.chunk_mb * 1024 * 1024
    total = probe(args.url)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.out.parent).free
    if not args.out.exists() and free < total * 1.2:
        sys.exit(f"not enough free disk: need {total * 1.2 / 1e9:.1f} GB, have {free / 1e9:.1f} GB.")
    parts = Path(str(args.out) + ".parts")
    done = set(json.loads(parts.read_text())) if parts.exists() and args.out.exists() else set()
    if not args.out.exists() or args.out.stat().st_size != total:
        with open(args.out, "wb") as f:
            f.truncate(total)
        done = set()
    n_chunks = -(-total // chunk)
    todo = [i for i in range(n_chunks) if i not in done]
    print(f"{total / 1e9:.2f} GB, {n_chunks} chunks, {len(done)} already done, "
          f"{args.connections} connections", flush=True)

    lock, counter, t0 = threading.Lock(), [0], time.time()
    with ThreadPoolExecutor(max_workers=args.connections) as pool:
        futures = [pool.submit(fetch_chunk, args.url, args.out, i, total, lock, done, parts, counter, chunk)
                   for i in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if k % 10 == 0 or k == len(futures):
                rate = counter[0] / max(time.time() - t0, 1e-9) / 1e6
                print(f"  {len(done)}/{n_chunks} chunks  {rate:.1f} MB/s", flush=True)

    if args.md5:
        h = hashlib.md5()
        with open(args.out, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        want = args.md5.split(":")[-1]
        if h.hexdigest() != want:
            sys.exit(f"CHECKSUM MISMATCH: got {h.hexdigest()}, expected {want}. The file is not trustworthy.")
        print("md5 verified", flush=True)
    parts.unlink(missing_ok=True)
    print(f"done: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
