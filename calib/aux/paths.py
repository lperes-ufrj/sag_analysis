"""Repository path anchors.

All input/output paths derive from the location of this file, not from the
cwd: the runners can be called from anywhere (python deconvolve/deconvolve.py
or cd deconvolve && python deconvolve.py).
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

CALIB_DIR  = REPO_ROOT / "calib"
DECONV_DIR = REPO_ROOT / "deconvolve"
FIT_DIR    = REPO_ROOT / "fit"

# .roots stays where it is today (calib/.roots); to be revisited
ROOTS_DIR = CALIB_DIR / ".roots"


def add_repo_to_path():
    """Allow 'import calib.aux...' when running any script directly."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def out(module_dir, kind, *parts):
    """<module_dir>/output/<kind>/<parts...>, creating the directories.

    kind: "plots" or "data".
    """
    p = Path(module_dir) / "output" / kind
    for part in parts:
        p = p / str(part)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_waveform_txt(path, wf):
    """One value per line, %.9e -- same format as the templates in
    filter/templates_large_pulses/."""
    np.savetxt(path, np.asarray(wf), fmt="%.9e")
    return path


def load_waveform_txt(path):
    return np.loadtxt(path)
