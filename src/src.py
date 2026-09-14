import numpy as np
from pathlib import Path
import re

def calc_integral_and_error(waveform, y_err):
    """Return (I - sigma, I, I + sigma) for uncorrelated sample errors.

    x is treated as exact; y_err contains standard errors on y. Trapezoidal
    endpoint and interior weights also support nonuniform sample spacing.
    """
    y = waveform[50:500]

    integral_central = np.trapezoid(y)

    integral_lower = np.trapezoid(y-y_err)
    integral_upper = np.trapezoid(y+y_err)

    return integral_lower, integral_central, integral_upper

def read_txt_error_file(sample_label,channels):
    """Return {run: {channel: baseline noise in ADC}} (not error on the mean)."""
    repo_dir = Path(__file__).resolve().parent.parent
    path_waveforms = (repo_dir / "coincidence/selected_waveforms" / sample_label)

    txt_suffix = f"{sample_label}.txt"

    txt_files = sorted(
        path_waveforms.glob(f"selection_cuts_coincidence_scan_run_*_{txt_suffix}")
    )
    
    y_err_runs = {}
    runs_files = set()
    for txt_file in txt_files:
        match = re.search(r"selection_cuts_coincidence_scan_run_(\d+)", txt_file.name)
        if match:
            run = int(match.group(1))
            runs_files.add((run, txt_file))

    for run in sorted(runs_files):
        with open(run[1], 'r') as f:
            text = f.read()
        matches = re.findall(r"Channel\s+(\d+):\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+ADC",text)

        y_err_run = {int(channel): float(error) for channel, error in matches}
        y_err_runs[run[0]] = y_err_run
    return y_err_runs
