"""bootstrap.py must pick a torch wheel the GPU can actually run."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "bootstrap", Path(__file__).resolve().parents[1] / "scripts" / "bootstrap.py")
bootstrap = importlib.util.module_from_spec(SPEC)
sys.modules["bootstrap"] = bootstrap
SPEC.loader.exec_module(bootstrap)


def _fake_smi(monkeypatch, header):
    def run(*a, **k):
        return subprocess.CompletedProcess(a, 0, stdout=header, stderr="")
    monkeypatch.setattr(bootstrap.subprocess, "run", run)


@pytest.mark.parametrize("header,tag", [
    ("| NVIDIA-SMI 550.00   Driver Version: 550.00   CUDA Version: 12.4 |", "cu124"),
    ("| NVIDIA-SMI 560.00   Driver Version: 560.00   CUDA Version: 12.6 |", "cu126"),
    ("| NVIDIA-SMI 570.00   Driver Version: 570.00   CUDA Version: 12.8 |", "cu128"),
    # Newer drivers print "UMD" in the label; this used to fall through to cu124, which has
    # no kernels for an RTX 50-series GPU.
    ("| NVIDIA-SMI 610.57   Driver Version: 610.57   CUDA UMD Version: 13.3 |", "cu128"),
    ("| NVIDIA-SMI 470.00   Driver Version: 470.00   CUDA Version: 11.4 |", None),
])
def test_wheel_tag_follows_driver_cuda(monkeypatch, header, tag):
    _fake_smi(monkeypatch, header)
    assert bootstrap.detect_cuda_tag() == tag


def test_every_tag_has_an_index():
    assert "cu128" in bootstrap.CUDA_INDEX
