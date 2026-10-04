#!/usr/bin/env python
"""Join uploaded parts back into a tar and unpack it, after verifying every checksum.

    python scripts/assemble_parts.py ~/workspace/uploads data/processed

The counterpart of scripts/upload_parts.py. Reads `SHA256SUMS` in the parts folder, refuses if any
part is missing or wrong, then streams the parts in name order into `tar` extraction under the
destination. Parts are left in place; delete the folder yourself once the result is checked.
"""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
from pathlib import Path


class Joined(io.RawIOBase):
    """Read-only stream over several files, back to back."""

    def __init__(self, files: list[Path]):
        self._files, self._i, self._fh = files, 0, None

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while True:
            if self._fh is None:
                if self._i >= len(self._files):
                    return 0
                self._fh = open(self._files[self._i], "rb")
                self._i += 1
            n = self._fh.readinto(b)
            if n:
                return n
            self._fh.close()
            self._fh = None


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    folder, dest = Path(sys.argv[1]).expanduser(), Path(sys.argv[2])
    sums = folder / "SHA256SUMS"
    if not sums.exists():
        sys.exit(f"{sums} not found: the upload is not finished.")
    files = []
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        part = folder / name.strip().lstrip("*")
        if not part.exists():
            sys.exit(f"missing part {part.name}")
        h = hashlib.sha256()
        with open(part, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != digest:
            sys.exit(f"CHECKSUM MISMATCH in {part.name}; resend it.")
        files.append(part)
    files.sort(key=lambda p: p.name)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BufferedReader(Joined(files)), mode="r|") as tar:
        tar.extractall(dest)
    print(f"verified {len(files)} parts and unpacked into {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
