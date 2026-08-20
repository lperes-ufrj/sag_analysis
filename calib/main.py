"""SPE calibration: fingerplot -> multigauss fit -> SPE template.

Usage:
    python calib/main.py            (or: cd calib && python main.py)

Outputs under calib/output/:
    plots/fingerplot/<run>/        raw fingerplot per channel
    plots/multigauss/<run>/        multigauss fit (SPE gain)
    plots/spe_template/<run>/<ch>/ persistences and template
    data/templates/<run>/template_<ch>.npy   <- consumed by deconvolve/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uproot

from calib.aux import paths
from calib.aux import fit_fingerplot as fg_fit
from calib.aux import spe_template as spe


def main():
    calib_run = str(input("Enter calibration run number: "))

    # read selected wfs from read_waveforms
    filename = paths.ROOTS_DIR / "metrics_select" / f"run{calib_run}_metrics_select.root"
    f = uproot.open(filename)

    # fingerplot multigauss fit -> {channel: Minuit}
    m_dict = fg_fit.fit_multigauss(f, calib_run)

    # build SPE template -> {channel: template_array}
    templates = spe.spe_template(f, m_dict, calib_run)

    print(f"\nTemplates saved to "
          f"{paths.CALIB_DIR / 'output' / 'data' / 'templates' / calib_run}")
    return m_dict, templates


if __name__ == "__main__":
    main()
