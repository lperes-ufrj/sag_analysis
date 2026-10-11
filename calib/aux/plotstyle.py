"""DUNE plot style, applied without any watermark.

The official style lives in https://github.com/DUNE/dune-plot-style and is not
on PyPI:

    pip install git+https://github.com/DUNE/dune-plot-style.git

Importing dunestyle.matplotlib is what applies the style. The watermark
functions (dunestyle.WIP(), .Preliminary(), .Official()) are deliberately NOT
called anywhere -- figures come out clean.

If dunestyle is not installed, a fallback approximating it is applied instead,
so the scripts still run and still look paper-like.
"""
import matplotlib as mpl

_FALLBACK = {
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.labelsize": "xx-large",
    "axes.titlesize": "large",
    "axes.linewidth": 1.2,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
    "legend.frameon": False,
    "legend.fontsize": "medium",
    "figure.autolayout": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def use_dune_style():
    """Apply the DUNE style. Returns True if the real package was used."""
    try:
        import dunestyle.matplotlib as dunestyle   # noqa: F401  (applies on import)
        found = True
    except ImportError:
        mpl.rcParams.update(_FALLBACK)
        print("dunestyle not installed -- using the fallback style "
              "(pip install git+https://github.com/DUNE/dune-plot-style.git)")
        found = False

    # no watermark, and a couple of things we want either way
    mpl.rcParams.update({"savefig.dpi": 200, "savefig.bbox": "tight"})
    return found
