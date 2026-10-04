#!/usr/bin/env python
"""Upload a folder of file parts to a remote host over ssh/scp, resumably, and verify them there.

    python scripts/upload_parts.py data/.cache/upload ai-gpu workspace/uploads

Meant for a slow link: the parts are sent one at a time, each to `<name>.partial` and renamed only
when complete, and a part whose remote size already matches is skipped, so rerunning after any
interruption continues where it stopped. `SHA256SUMS` is sent last and checked on the remote side
with `sha256sum -c`; a part that fails the check is deleted so the next run resends it. Only needs
`ssh` and `scp` on the PATH (any host alias from your ssh config works, including one that goes
through a tunnel).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes", "-o", "ConnectTimeout=30",
            "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=6"]


def ssh(host: str, command: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH_OPTS, host, command], capture_output=True, text=True, timeout=timeout)


def remote_sizes(host: str, remote_dir: str) -> dict[str, int]:
    out = ssh(host, f"mkdir -p {remote_dir} && cd {remote_dir} && stat -c '%n %s' * 2>/dev/null; true")
    sizes = {}
    for line in out.stdout.splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            sizes[parts[0]] = int(parts[1])
    return sizes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("host", help="ssh host alias")
    ap.add_argument("remote_dir", help="directory on the host, relative to its home")
    ap.add_argument("--retries", type=int, default=20)
    args = ap.parse_args()

    files = sorted(p for p in args.folder.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    if not files or not (args.folder / "SHA256SUMS").exists():
        sys.exit(f"{args.folder} needs the parts and a SHA256SUMS file.")
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} parts, {total / 1e6:.0f} MB to {args.host}:{args.remote_dir}", flush=True)

    for attempt in range(args.retries):
        try:
            have = remote_sizes(args.host, args.remote_dir)
            todo = [p for p in files if have.get(p.name) != p.stat().st_size]
            print(f"attempt {attempt + 1}: {len(files) - len(todo)}/{len(files)} parts already there", flush=True)
            for p in todo:
                t0 = time.time()
                r = subprocess.run(["scp", "-q", *SSH_OPTS, str(p), f"{args.host}:{args.remote_dir}/{p.name}.partial"])
                if r.returncode != 0:
                    raise RuntimeError(f"scp of {p.name} failed ({r.returncode})")
                ssh(args.host, f"cd {args.remote_dir} && mv {p.name}.partial {p.name}")
                print(f"  sent {p.name} {p.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
            r = subprocess.run(["scp", "-q", *SSH_OPTS, str(args.folder / "SHA256SUMS"),
                                f"{args.host}:{args.remote_dir}/SHA256SUMS"])
            if r.returncode != 0:
                raise RuntimeError("scp of SHA256SUMS failed")
            check = ssh(args.host, f"cd {args.remote_dir} && sha256sum -c SHA256SUMS 2>&1", timeout=600)
            bad = [ln.split(":")[0] for ln in check.stdout.splitlines() if "FAILED" in ln]
            if bad:
                print(f"  checksum failed for {bad}; deleting and resending", flush=True)
                ssh(args.host, f"cd {args.remote_dir} && rm -f {' '.join(bad)}")
                continue
            if check.returncode == 0:
                print("ALL PARTS VERIFIED", flush=True)
                return 0
            raise RuntimeError(f"sha256sum -c exited {check.returncode}: {check.stdout[-200:]}")
        except Exception as exc:  # a dropped tunnel or timeout is routine on a long upload
            print(f"  problem: {exc}; retrying in 30 s", flush=True)
            time.sleep(30)
    sys.exit("gave up after repeated failures; rerun to continue from where it stopped.")


if __name__ == "__main__":
    sys.exit(main())
