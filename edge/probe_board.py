#!/usr/bin/env python3
"""Report the facts about a Jetson that decide everything downstream.

    python3 edge/probe_board.py
    python3 edge/probe_board.py --json > board.json

Run this ON the board and paste the output back. It answers FD-003: which module,
which JetPack, which TensorRT, which power modes, what cooling headroom. Nothing
else (export opset, precision strategy, what to benchmark) should be decided until
these are recorded in docs/hardware.md.

Standard library only, and written for Python 3.8, because edge/ must not import
training dependencies and JetPack 5.x ships an old interpreter. Every probe is
independent and fails soft -- run on a laptop it reports "not found" for each item
rather than crashing, which is also how you tell you are not on the board.

Does NOT need sudo. The power-mode list is read from /etc/nvpmodel.conf rather than
queried, which is also the authoritative test for Super mode: a module running
JetPack 6.2+ lists a MAXN_SUPER mode; one that does not, does not.
"""

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys


def sh(cmd, timeout=15):
    """Run a command, return stripped stdout or None. Never raises."""
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, timeout=timeout)
        # A failed command must read as "nothing", never as its error text. Returning
        # stderr made `dpkg-query ... libnvinfer8` look like a value, so the `or`
        # fallback to libnvinfer10 never ran -- found only by running on a real board.
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:
        return None


def read(path):
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read().strip().strip("\x00")
    except Exception:
        return None


def probe_module():
    model = read("/proc/device-tree/model")
    return {"device_tree_model": model, "is_jetson": bool(model and "jetson" in model.lower()
                                                           or (model and "orin" in model.lower()))}


def probe_l4t():
    raw = read("/etc/nv_tegra_release")
    info = {"raw": raw}
    if raw:
        m = re.search(r"R(\d+)\s*\(release\),\s*REVISION:\s*([\d.]+)", raw)
        if m:
            info["l4t"] = "{}.{}".format(m.group(1), m.group(2))
    return info


def probe_jetpack():
    ver = sh(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"])
    return {"nvidia_jetpack_package": ver if ver and "no packages" not in ver.lower() else None}


def probe_tensorrt():
    info = {}
    code = "import tensorrt; print(tensorrt.__version__)"
    info["python_module"] = sh([sys.executable, "-c", code])
    if info["python_module"] and "Error" in info["python_module"]:
        info["python_module"] = None
    info["libnvinfer_package"] = sh(["dpkg-query", "-W", "-f=${Version}", "libnvinfer8"]) \
        or sh(["dpkg-query", "-W", "-f=${Version}", "libnvinfer10"])
    trtexec = shutil.which("trtexec") or ("/usr/src/tensorrt/bin/trtexec"
                                          if os.path.exists("/usr/src/tensorrt/bin/trtexec") else None)
    info["trtexec"] = trtexec
    return info


def probe_cuda():
    info = {}
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc"
                                    if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
    if nvcc:
        out = sh([nvcc, "--version"]) or ""
        m = re.search(r"release ([\d.]+)", out)
        info["nvcc"] = m.group(1) if m else out[-80:]
    info["cuda_version_json"] = read("/usr/local/cuda/version.json")
    return info


def probe_power_modes():
    """Mode names come from the config, so no sudo is needed."""
    conf = read("/etc/nvpmodel.conf")
    modes = []
    if conf:
        for m in re.finditer(r"<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=([^\s>]+)\s*>", conf):
            modes.append({"id": int(m.group(1)), "name": m.group(2)})
    current = sh(["nvpmodel", "-q"])
    names = [m["name"] for m in modes]
    return {
        "modes": modes,
        "current_query": current,
        "has_maxn_super": any("SUPER" in n.upper() for n in names),
        "has_25w": any("25W" in n.upper() for n in names),
    }


def probe_memory():
    mem = read("/proc/meminfo") or ""
    def kb(key):
        m = re.search(r"^%s:\s+(\d+)\s+kB" % key, mem, re.M)
        return int(m.group(1)) if m else None
    total, avail, swap = kb("MemTotal"), kb("MemAvailable"), kb("SwapTotal")
    return {
        "mem_total_gb": round(total / 1048576, 2) if total else None,
        "mem_available_gb": round(avail / 1048576, 2) if avail else None,
        "swap_total_gb": round(swap / 1048576, 2) if swap is not None else None,
        "unified_memory_note": "CPU and GPU share this pool; the OS and desktop use some "
                               "before any process starts.",
    }


def probe_storage():
    src = sh(["findmnt", "-no", "SOURCE", "/"])
    kind = None
    if src:
        kind = ("NVMe" if "nvme" in src else "SD card / eMMC" if "mmcblk" in src
                else "USB / SATA" if re.search(r"/dev/sd", src) else "unknown")
    stat = None
    try:
        du = shutil.disk_usage("/")
        stat = {"total_gb": round(du.total / 1e9, 1), "free_gb": round(du.free / 1e9, 1)}
    except Exception:
        pass
    return {"root_device": src, "kind": kind, "usage": stat,
            "note": "SD-card boot will bottleneck video decode; prefer NVMe."}


def probe_thermal():
    zones = {}
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        name, temp = read(z + "/type"), read(z + "/temp")
        if name and temp and temp.lstrip("-").isdigit():
            zones[name] = round(int(temp) / 1000.0, 1)
    fan = bool(glob.glob("/sys/devices/platform/pwm-fan*") or glob.glob("/sys/class/hwmon/*/pwm1"))

    # An interface existing does not mean a fan is fitted: the kernel driver is present
    # on boards with no fan connected. The tachometer is the evidence -- it reads zero
    # when nothing is spinning. Report both so neither is mistaken for the other.
    rpm = None
    for f in glob.glob("/sys/class/hwmon/hwmon*/rpm"):
        val = read(f)
        if val and val.isdigit():
            rpm = int(val)
            break
    duty = None
    for f in glob.glob("/sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1"):
        val = read(f)
        if val and val.isdigit():
            duty = int(val)
            break
    return {"zones_c": zones, "fan_interface_present": fan,
            "fan_rpm": rpm, "fan_pwm_duty_of_255": duty,
            "fan_spinning": bool(rpm and rpm > 0),
            "note": "Record AMBIENT temperature yourself. Even with an active fan, burst "
                    "FPS is not deployment FPS: log temperature and clocks on every "
                    "sustained run."}


def collect():
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "module": probe_module(),
        "l4t": probe_l4t(),
        "jetpack": probe_jetpack(),
        "tensorrt": probe_tensorrt(),
        "cuda": probe_cuda(),
        "power_modes": probe_power_modes(),
        "memory": probe_memory(),
        "storage": probe_storage(),
        "thermal": probe_thermal(),
    }


def verdict(d):
    """Plain-language reading of the facts, with the uncertainty left in."""
    lines = []
    if not d["module"]["is_jetson"]:
        lines.append("NOT A JETSON: nothing below describes the target board. "
                     "Run this on the Orin Nano itself.")
        return lines

    pm = d["power_modes"]
    if pm["modes"]:
        lines.append("Power modes available: " + ", ".join(m["name"] for m in pm["modes"]))
        if pm["has_maxn_super"]:
            lines.append("SUPER MODE: available (a MAXN_SUPER mode is listed).")
        else:
            lines.append("SUPER MODE: not available. No MAXN_SUPER mode is listed, which "
                         "means this is not running JetPack 6.2 or later.")
    else:
        lines.append("Could not read /etc/nvpmodel.conf, so the power modes and Super "
                     "status are UNKNOWN. Do not guess; run `sudo nvpmodel -q --verbose`.")

    if d["l4t"].get("l4t"):
        lines.append("L4T %s. Record this and freeze it: an upgrade mid-study "
                     "invalidates every engine and every number collected." % d["l4t"]["l4t"])
    else:
        lines.append("L4T version could not be read.")

    if not d["tensorrt"]["python_module"] and not d["tensorrt"]["libnvinfer_package"]:
        lines.append("TensorRT: not detected. JetPack normally installs it; the build "
                     "chain cannot proceed without it.")

    st = d["storage"]
    if st["kind"] and "SD" in st["kind"]:
        lines.append("STORAGE: booting from an SD card. Expect video decode to be "
                     "storage-limited; note this before blaming the model.")
    th = d["thermal"]
    if th["fan_spinning"]:
        lines.append("COOLING: active fan, spinning at %s RPM (PWM duty %s/255). Sustained "
                     "load is still the real test: the fan curve and heatsink decide "
                     "whether higher modes throttle." % (th["fan_rpm"], th["fan_pwm_duty_of_255"]))
    elif th["fan_interface_present"]:
        lines.append("COOLING: a fan driver exists but the tachometer reads %s, so no fan "
                     "is confirmed spinning. Treat the board as passively cooled until "
                     "proven otherwise: the 30-minute sustained-load test is essential."
                     % th["fan_rpm"])
    else:
        lines.append("COOLING: no fan interface found. Treat the board as passively "
                     "cooled: the 30-minute sustained-load test is essential, and the "
                     "higher power modes may throttle quickly.")
    mem = d["memory"]
    if mem["mem_total_gb"]:
        lines.append("Memory: %.1f GB unified, %.1f GB currently available."
                     % (mem["mem_total_gb"], mem["mem_available_gb"] or 0))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output only")
    args = ap.parse_args()

    data = collect()
    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    print("=" * 62)
    print("  Jetson board probe")
    print("=" * 62)
    print(json.dumps(data, indent=2))
    print()
    print("=" * 62)
    print("  Reading")
    print("=" * 62)
    for line in verdict(data):
        print("  * " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
