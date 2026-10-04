"""Read zip archives whose members use Deflate64 (compression method 9).

Python's `zipfile` cannot decompress Deflate64, which some Windows tools and data hosts use
(the Kakadu archive on Zenodo is one: every member is method 9). The member list and offsets
are still readable with `zipfile`, so this reads the central directory with it and decompresses
the bytes itself, using `inflate64` for method 9. Every member is checked against the CRC-32 and
size recorded in the archive, so a corrupt download fails loudly instead of yielding bad images.
"""

from __future__ import annotations

import struct
import zipfile
import zlib
from pathlib import Path


def decompress(method: int, data: bytes) -> bytes:
    if method == zipfile.ZIP_STORED:
        return data
    if method == zipfile.ZIP_DEFLATED:
        return zlib.decompress(data, -15)
    if method == 9:
        try:
            import inflate64
        except ImportError as exc:
            raise SystemExit("Deflate64 archive: install the reader with `pip install inflate64` "
                             "(listed in requirements-train.txt).") from exc
        return inflate64.Inflater().inflate(data)
    raise NotImplementedError(f"zip compression method {method} is not supported")


class Zip64Reader:
    """Random access to the members of a zip archive, including Deflate64 ones."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if not self.path.exists():
            raise SystemExit(f"archive not found: {self.path}")
        self._zip = zipfile.ZipFile(self.path)
        self._info = {i.filename: i for i in self._zip.infolist() if not i.is_dir()}
        self._fh = open(self.path, "rb")

    def names(self) -> list[str]:
        return list(self._info)

    def __contains__(self, name: str) -> bool:
        return name in self._info

    def read(self, name: str) -> bytes:
        info = self._info[name]
        self._fh.seek(info.header_offset)
        header = self._fh.read(30)
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        self._fh.seek(info.header_offset + 30 + name_len + extra_len)
        out = decompress(info.compress_type, self._fh.read(info.compress_size))
        if len(out) != info.file_size or (zlib.crc32(out) & 0xFFFFFFFF) != info.CRC:
            raise ValueError(f"{name}: size or CRC does not match the archive; the file is corrupt")
        return out

    def close(self) -> None:
        self._fh.close()
        self._zip.close()

    def __enter__(self) -> "Zip64Reader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
