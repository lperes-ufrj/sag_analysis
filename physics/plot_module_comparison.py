"""Compara M5, M6 e/ou M8 (fitting-bertolini-avg.py) vs campo eletrico.

Escolha os modulos a comparar na linha de comando, por exemplo:
    python plot_module_comparison.py M5 M6 M8   # todos (tambem o default)
    python plot_module_comparison.py M5 M6
    python plot_module_comparison.py M8

Para tau_slow e f_fast gera 3 pngs em plots/<modulos>/ (a pasta muda
conforme a combinacao escolhida, ex. plots/M5-M6-M8/, plots/M5-M6/,
plots/M8/): *_raw.png (curvas cru), *_peaknorm.png (cada curva normalizada
pelo seu proprio pico) e *_mean.png (media ponto-a-ponto, com incerteza =
erro padrao da media entre modulos; para tau_slow tambem ajusta o modelo de
Segreto 2021, Eq. 14, igual ao fit em fitting-bertolini-avg.py).

I_f - I_s so tem 2 pngs (nao existe versao "raw"): a diferenca crua mistura
as escalas absolutas dos dois componentes (I_s domina em valor absoluto) e
da uma curva sempre negativa, sem significado fisico. O jeito certo, igual
ao I_dif de fitting-bertolini-avg.py, e normalizar CADA componente pelo seu
proprio pico primeiro (r_f = I_f/I_f_peak, r_s = I_s/I_s_peak) e SO DEPOIS
subtrair: r_f - r_s. Isso vale 0 em E = 0 por construcao e fica positivo (a
componente rapida cai menos que a lenta com o campo). idiff_peaknorm.png e
idiff_mean.png usam essa versao normalizada.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle
from iminuit import Minuit
from iminuit.cost import LeastSquares

# caminhos ancorados no proprio script: funciona de qualquer diretorio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # sag_analysis/physics
REPO_DIR = os.path.dirname(SCRIPT_DIR)                    # sag_analysis
FIT_DIR = os.path.join(REPO_DIR, 'fitting')               # saidas do fitting-bertolini-avg.py

ALL_MODULES = ['M5', 'M6', 'M8']
MARKERS = {'M5': 'o', 'M6': 's', 'M8': '^'}
BASE = FIT_DIR + '/plots_bertolini_avg/{mod}/parameters/'


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('modules', nargs='*', metavar='MODULE',
                    help=f'modulos a comparar, entre {ALL_MODULES} (default: todos)')
    args = p.parse_args()
    args.modules = args.modules or ALL_MODULES
    bad = [m for m in args.modules if m not in ALL_MODULES]
    if bad:
        p.error(f'modulo(s) invalido(s): {bad} -- escolha entre {ALL_MODULES}')
    return args


def load(mod):
    base = BASE.format(mod=mod)
    par = np.genfromtxt(base + 'fit_params_vs_field.txt', names=True, comments='#')
    ing = np.genfromtxt(base + 'integrals_vs_field.txt', names=True, comments='#')
    return par, ing


def add_watermarks(ax, side='left'):
    """DUNE Preliminary + Work In Progress, empilhados no canto inferior
    (esquerdo ou direito, dentro dos eixos). fontsize menor que o padrao
    (18) do dunestyle para incomodar menos a legenda."""
    x = 0.05 if side == 'left' else 0.95
    align = side
    dunestyle.Preliminary(x=x, y=0.08, align=align, ax=ax, fontsize=11)
    dunestyle.WIP(x=x, y=0.03, align=align, ax=ax, fontsize=11)


# --- observaveis -----------------------------------------------------------

def tauslow(mod, data):
    par, ing = data[mod]
    return par['tau_slow'], par['tau_slow_err']


def fprompt(mod, data):
    par, ing = data[mod]
    return ing['f_fast'], ing['f_fast_err']


def normalize_by_peak(val, err):
    """r = val / val[pico], onde pico = ponto de maior |val|.

    O erro carrega o termo do denominador comum (o proprio pico e um dos
    pontos, com sua propria incerteza -- todo mundo divide por ele)."""
    ipeak = np.argmax(np.abs(val))
    peak, peak_err = val[ipeak], err[ipeak]
    r = val / peak
    r_err = np.abs(r) * np.hypot(err / val, peak_err / peak)
    r_err[ipeak] = 0.0
    return r, r_err


def tauslow_peaknorm(mod, data):
    return normalize_by_peak(*tauslow(mod, data))


def fprompt_peaknorm(mod, data):
    return normalize_by_peak(*fprompt(mod, data))


def idiff_peaknorm(mod, data):
    """r_f - r_s, cada componente normalizada pelo seu proprio pico
    (equivalente ao I_dif de fitting-bertolini-avg.py, que usa E=0 --
    aqui o pico de I_fast e I_slow tambem cai em E=0)."""
    par, ing = data[mod]
    r_fast, r_fast_err = normalize_by_peak(ing['I_fast'], ing['I_fast_err'])
    r_slow, r_slow_err = normalize_by_peak(ing['I_slow'], ing['I_slow_err'])
    return r_fast - r_slow, np.hypot(r_fast_err, r_slow_err)


# --- plots -------------------------------------------------------------

def plot_overlay(modules, data, E, getter, ylabel, outpath,
                  legend_loc='best', hlines=(), watermark_side='left'):
    fig, ax = plt.subplots()
    for mod in modules:
        val, err = getter(mod, data)
        ax.errorbar(E, val, yerr=err, fmt=MARKERS[mod], ms=4, label=mod)
    for y in hlines:
        ax.axhline(y, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Electric Field [kV/cm]')
    ax.set_ylabel(ylabel)
    add_watermarks(ax, side=watermark_side)
    ax.legend(loc=legend_loc)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def mean_and_unc(modules, data, getter):
    vals = np.array([getter(mod, data)[0] for mod in modules])
    mean = vals.mean(axis=0)

    if len(modules) > 1:
        unc = vals.std(axis=0, ddof=1) / np.sqrt(len(modules))
        unc_label = r'$\langle\sigma\rangle$' + f' = {np.mean(unc):.3g}'
    else:
        unc = getter(modules[0], data)[1]
        unc_label = 'unico modulo, erro do ajuste'

    return mean, unc, unc_label


def plot_mean(modules, data, E, getter, ylabel, outpath, legend_loc='best',
              watermark_side='left'):
    mean, unc, unc_label = mean_and_unc(modules, data, getter)

    fig, ax = plt.subplots()
    ax.errorbar(E, mean, yerr=unc, fmt='o', ms=4, color='k',
                label=f'Mean ({", ".join(modules)})\n{unc_label}')
    ax.set_xlabel('Electric Field [kV/cm]')
    ax.set_ylabel(ylabel)
    add_watermarks(ax, side=watermark_side)
    ax.legend(loc=legend_loc)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# tau_slow(E) -- Segreto 2021 (PRD 103, 043001), Eq. 14, igual a
# fitting-bertolini-avg.py: beta/(1 + k_E/E) reescrito como beta*E/(E+k_E),
# identico para E > 0 e finito em E = 0 (onde tau_eff = 1/alpha)
def tau_eff_segreto(E, alpha, beta, k_E):
    return 1.0 / (alpha + beta * E / (E + k_E))


def fit_segreto(E, tau, tau_err):
    m = Minuit(LeastSquares(E, tau, tau_err, tau_eff_segreto),
               alpha=1.0 / tau[0], beta=0.2, k_E=0.075)
    m.limits['alpha'] = (0.1, 5.0)
    m.limits['beta'] = (0, 5.0)
    m.limits['k_E'] = (0.001, 2.0)
    m.migrad()
    m.hesse()
    return m


def plot_mean_tauslow_segreto(modules, data, E, outpath, legend_loc='best'):
    mean, unc, unc_label = mean_and_unc(modules, data, tauslow)

    m = fit_segreto(E, mean, unc)
    ndof = len(E) - m.nfit
    chi2_ndof = m.fval / ndof

    alpha, beta, k_E = m.values['alpha'], m.values['beta'], m.values['k_E']
    tau_0 = 1.0 / alpha
    tau_inf = 1.0 / (alpha + beta)

    print('\n=== tau_slow(E), media '
          f'({", ".join(modules)}): Segreto 2021, Eq. 14 ===')
    print(m)
    print(f'tau_eff(0)   = {tau_0 * 1000:.1f} ns')
    print(f'tau_eff(inf) = {tau_inf * 1000:.1f} ns   '
          f'({(tau_inf / tau_0 - 1) * 100:+.1f}%)')
    print(f'k_E = {k_E:.4f} +- {m.errors["k_E"]:.4f} kV/cm '
          '[Segreto: 0.075 +- 0.015]')
    print(f'chi2/ndof = {chi2_ndof:.2f}')

    E_smooth = np.linspace(0, E.max() * 1.05, 400)

    fig, ax = plt.subplots()
    ax.errorbar(E, mean, yerr=unc, fmt='o', ms=4, color='k',
                label=f'Mean ({", ".join(modules)})\n{unc_label}')
    ax.plot(E_smooth, tau_eff_segreto(E_smooth, alpha, beta, k_E), 'r-', lw=1.2,
            label=(rf'$\tau_{{eff}}(0)$ = {tau_0 * 1000:.0f} ns' '\n'
                   rf'$\tau_{{eff}}(\infty)$ = {tau_inf * 1000:.0f} ns' '\n'
                   rf'$k_\mathcal{{E}}$ = {k_E:.3f} $\pm$ {m.errors["k_E"]:.3f} kV/cm' '\n'
                   rf'$\chi^2$/ndof = {chi2_ndof:.1f}'))
    ax.set_xlabel('Electric Field [kV/cm]')
    ax.set_ylabel(r'$\tau_{slow}$ [$\mu$s]')
    add_watermarks(ax)
    ax.legend(loc=legend_loc, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    modules = [m for m in ALL_MODULES if m in args.modules]

    data = {mod: load(mod) for mod in modules}
    E = data[modules[0]][1]['efield_kVcm']
    assert all(np.allclose(data[m][1]['efield_kVcm'], E) for m in modules)

    outdir = os.path.join(SCRIPT_DIR, f'plots/{"-".join(modules)}/')
    os.makedirs(outdir, exist_ok=True)

    # --- I_fast - I_slow (so a versao normalizada -- ver nota no topo) ---
    plot_overlay(modules, data, E, idiff_peaknorm,
                 r'$I_f/I_f^{\rm peak} - I_s/I_s^{\rm peak}$',
                 outdir + 'idiff_peaknorm.png', hlines=(0.0,),
                 watermark_side='right')
    plot_mean(modules, data, E, idiff_peaknorm,
              r'$I_f/I_f^{\rm peak} - I_s/I_s^{\rm peak}$',
              outdir + 'idiff_mean.png', watermark_side='right')

    # --- tau_slow --------------------------------------------------------
    plot_overlay(modules, data, E, tauslow, r'$\tau_{slow}$ [$\mu$s]',
                 outdir + 'tauslow_raw.png')
    plot_overlay(modules, data, E, tauslow_peaknorm,
                 r'$\tau_{slow}\,/\,\tau_{slow}^{\rm peak}$',
                 outdir + 'tauslow_peaknorm.png', hlines=(1.0,))
    plot_mean_tauslow_segreto(modules, data, E, outdir + 'tauslow_mean.png')

    # --- f-prompt --------------------------------------------------------
    plot_overlay(modules, data, E, fprompt, r'$f_{prompt}$',
                 outdir + 'fprompt_raw.png', watermark_side='right')
    plot_overlay(modules, data, E, fprompt_peaknorm,
                 r'$f_{prompt}\,/\,f_{prompt}^{\rm peak}$',
                 outdir + 'fprompt_peaknorm.png', hlines=(1.0,),
                 watermark_side='right')
    plot_mean(modules, data, E, fprompt, r'$f_{prompt}$',
              outdir + 'fprompt_mean.png', watermark_side='right')

    print('done ->', outdir)


if __name__ == '__main__':
    main()
