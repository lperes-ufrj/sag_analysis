import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calib.aux.runs import runs_of

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT    = REPO_ROOT / "deconvolve" / "deconvolve_coincidence.py"
COINC_DIR = REPO_ROOT / "coincidence" / "selected_waveforms"
CHANNELS  = (2080, 2081)
SUFFIX    = ("coinc_2030-2031-2040-2041_vs_2050-2051-2060-2061"
             "_save_2080-2081_window_10_ticks_min_amplitude_0_adc")

# run map lives in calib/aux/runs.py (shared with fit/)
runs = {f"{kv} kV": int(run) for kv, run in runs_of("up")}

template = str(input("Which template ('francesca' or 'mine')? "))

n_ok = n_fail = 0
for label, run in runs.items():
    for channel in CHANNELS:
        # the aggregate CSV (one per channel), not the coincidence index;
        # run numbers are zero-padded to 6 digits in the filenames
        file_name = COINC_DIR / f"channel_{channel}_run_{run:06d}_{SUFFIX}.csv"

        if not file_name.is_file():
            print(f"[MISSING] {label:>8}  run {run:06d}  ch {channel}  ->  {file_name.name}")
            n_fail += 1
            continue

        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(file_name), template],
            capture_output=True, text=True)

        if result.returncode == 0:
            print(f"[OK]      {label:>8}  run {run:06d}  ch {channel}")
            for line in result.stdout.splitlines():
                if line.strip().startswith("M"):
                    print(f"          {line.strip()}")
            n_ok += 1
        else:
            print(f"[ERROR]   {label:>8}  run {run:06d}  ch {channel}")
            print("          " + (result.stderr.strip().splitlines() or ["(no stderr)"])[-1])
            n_fail += 1

print(f"\n{n_ok} OK, {n_fail} failed")
