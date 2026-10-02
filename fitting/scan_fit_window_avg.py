"""tau_slow (E = 0) em funcao da janela de ajuste, para M5/M6/M8.

Usa os dados combinados dos dois canais de cada modulo (igual a
fitting-bertolini-avg.py: media dos dois espectros, kernel = media dos dois
templates), diferente do scan_fitmax.py, que roda um canal so (M8, 2080).
Script separado porque aqui da pra comparar os 3 modulos.

Tres estudos:

1. tau_slow(FIT_MAX), FIT_MIN fixo em 0.25 us -- mesma ideia do
   scan_fitmax.py original: estica o teto da janela ate 5 us (bem alem do
   FIT_MAX = 3 us padrao) pra ver se a cauda lenta esta sendo truncada. Um
   png por modulo, mais um com os 3 sobrepostos. Foi daqui que saiu o
   FIT_MAX por arapuca agora em uso no fitting-bertolini-avg.py (M8 = 2.95
   us, M5 = M6 = 2.56 us -- M6 ainda sem scan proprio, usa o do M5).

1b. tau_slow(FIT_MIN), agora com FIT_MAX FIXO no valor de producao acima
    (por modulo) e FIT_MIN variando um pouco pra frente e pra tras do
    padrao (0.25 us). Mesmo formato do estudo 1, eixos trocados.

2. So para M6: a janela varrida nas DUAS pontas (FIT_MIN x FIT_MAX) --
   heatmap anotado de tau_slow e da integral I_slow = Aslow*tau_slow em
   funcao das duas, pra ver numericamente (nao so "a olho") o quanto a
   POSICAO da janela pesa no resultado, nao so o tamanho.

Saida em debug_scan_fitwindow/ -- estudo de sistematica, fora do pipeline
principal (fitting-bertolini-avg.py e quem gera os txt usados no resto da
analise).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import fftconvolve
from iminuit import Minuit
from iminuit.cost import LeastSquares

# caminhos ancorados no proprio script: funciona de qualquer diretorio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # sag_analysis/fitting
REPO_DIR = os.path.dirname(SCRIPT_DIR)                    # sag_analysis

SAMPLE = 16  # ns
time_array = np.arange(0, 1024 * SAMPLE, SAMPLE) / 1000  # us

RUN_0 = 39510   # campo nulo -- mesmo run para M5, M6 e M8 (ver runs{} em
                # fitting-bertolini-avg.py: a chave e o campo, nao o modulo)

# rodada de 09/set/2026 (mesma referencia de fitting-bertolini-avg.py)
SCAN_M8 = '20260909_125014'
SCAN_LOW = '20260909_213507'
PATTERN_NEW = 'channel_{ch}_coincidence_scan_run_0{run}_{scan}.csv'

MODULE_CONFIG = {
    'M8': dict(channels=(2080, 2081),
               templates=('template_42156_M8_1.txt', 'template_42156_M8_2.txt'),
               subdir=SCAN_M8),
    'M6': dict(channels=(2060, 2061),
               templates=('template_40808_M6_1.txt', 'template_40808_M6_2.txt'),
               subdir=SCAN_LOW),
    'M5': dict(channels=(2050, 2051),
               templates=('template_41522_M5_1.txt', 'template_41522_M5_2.txt'),
               subdir=SCAN_LOW),
}

MODULES = ['M5', 'M6', 'M8']
COLORS = {'M5': 'C0', 'M6': 'C1', 'M8': 'C2'}
MARKERS = {'M5': 'o', 'M6': 's', 'M8': '^'}

TAU_FAST_FIXED = 0.006   # us, fixo -- mesma justificativa de fitting-bertolini-avg.py
T0_CENTER = 1.008
T0_HALF = 0.010

LIMITS = {
    't0':       (T0_CENTER - T0_HALF, T0_CENTER + T0_HALF),
    'Aslow':    (0, 5),
    'Afast':    (0, 100),
    # teto largo (2.5 us, vs 1.6 us do ajuste padrao): o objetivo aqui e
    # justamente ver ate onde tau_slow quer ir quando a janela cresce, sem
    # o teto padrao mascarando isso
    'tau_slow': (0.7, 2.5),
    'tau_fast': (0.001, 0.05),
}

INITIAL_GUESSES = dict(t0=T0_CENTER, Aslow=1.16, Afast=17.1, tau_slow=1.4,
                        tau_fast=TAU_FAST_FIXED)

OUTDIR = os.path.join(SCRIPT_DIR, 'debug_scan_fitwindow') + '/'
os.makedirs(OUTDIR, exist_ok=True)


# --- dados e modelo ----------------------------------------------------

def coincidence_file(cfg, ch, run):
    return (REPO_DIR + f'/coincidence/selected_waveforms/{cfg["subdir"]}/'
            + PATTERN_NEW.format(ch=ch, run=run, scan=cfg['subdir']))


def make_causal_kernel(template, n_baseline=5):
    """Copia literal de fitting-bertolini-avg.py: subtrai baseline, leva a
    subida pro indice 0, normaliza area unitaria."""
    k = np.nan_to_num(np.asarray(template, dtype=float), nan=0.0)

    rise = int(np.argmax(np.abs(np.diff(k))))

    if rise > n_baseline:
        k = k - np.median(k[:rise - n_baseline])

    k = np.roll(k, -rise)

    if rise > 0:
        k[-rise:] = 0.0

    dt = SAMPLE / 1000.0

    return k / (k.sum() * dt)


def make_model(kernel):
    """Fecha o kernel do modulo numa funcao de modelo propria -- evita
    depender de um FIT_KERNEL global mutavel quando varios modulos (cada
    um com seu kernel) entram no mesmo script."""

    def model(t, t0, Aslow, Afast, tau_slow, tau_fast):
        dt = time_array[1] - time_array[0]

        I = np.zeros_like(time_array)
        mask = time_array >= t0
        I[mask] = (Aslow * np.exp(-(time_array[mask] - t0) / tau_slow)
                   + Afast * np.exp(-(time_array[mask] - t0) / tau_fast))

        full_model = fftconvolve(I, kernel, mode='full')[:len(time_array)] * dt

        return np.interp(t, time_array, full_model)

    return model


def load_module_waveform(mod):
    """Waveform media dos 2 canais do modulo em E = 0 (normalizada pelo
    pico), sigma de ruido (std da cauda t > 8 us) e o kernel de ajuste
    (media dos kernels dos 2 templates do modulo)."""
    cfg = MODULE_CONFIG[mod]

    dfs = [pd.read_csv(coincidence_file(cfg, ch, RUN_0)) for ch in cfg['channels']]
    waveform = np.mean([df['mean'].to_numpy() for df in dfs], axis=0)
    waveform = waveform / waveform.max()

    templates = [np.loadtxt(
        REPO_DIR + '/calib/output/data/templates/francesca_template/' + t
    ) for t in cfg['templates']]
    kernel = 0.5 * sum(make_causal_kernel(t) for t in templates)

    noise_region = time_array > 8
    sigma = np.std(waveform[noise_region] - np.mean(waveform[noise_region]))

    return waveform, sigma, kernel


def at_limit(m, name):
    lo, hi = m.limits[name]
    return abs(m.values[name] - lo) < 1e-6 or abs(m.values[name] - hi) < 1e-6


def fit_window(model, waveform, sigma, fit_min, fit_max, seed):
    mask = (time_array > fit_min) & (time_array < fit_max)
    x = time_array[mask]
    y = waveform[mask]
    yerr = sigma * np.ones_like(y)

    m = Minuit(LeastSquares(x, y, yerr, model), **seed)

    for name, lim in LIMITS.items():
        m.limits[name] = lim

    m.fixed['tau_fast'] = True
    m.migrad()
    m.hesse()

    ok = bool(m.valid and not any(
        at_limit(m, p) for p in m.parameters if not m.fixed[p]))

    return m, ok


# --- estudo 1: tau_slow(FIT_MAX), por modulo e os 3 juntos -------------

FIT_MIN = 0.25                              # us -- mesmo valor do ajuste padrao
FITMAX_SCAN = np.arange(2.0, 5.001, 0.1)    # us -- bem alem do FIT_MAX = 3 us padrao

results = {}

for mod in MODULES:
    print(f'\n=== {mod}: varredura em FIT_MAX (FIT_MIN = {FIT_MIN} us) ===')

    waveform, sigma, kernel = load_module_waveform(mod)
    model = make_model(kernel)

    tau, tau_err, chi2, ok = [], [], [], []
    seed = dict(INITIAL_GUESSES)

    for fit_max in FITMAX_SCAN:
        m, is_ok = fit_window(model, waveform, sigma, FIT_MIN, fit_max, seed)

        n = int(((time_array > FIT_MIN) & (time_array < fit_max)).sum())

        tau.append(m.values['tau_slow'] * 1000)      # ns
        tau_err.append(m.errors['tau_slow'] * 1000)
        chi2.append(m.fval / (n - m.nfit))
        ok.append(is_ok)

        if m.valid:
            seed = m.values.to_dict()

    tau, tau_err = np.array(tau), np.array(tau_err)
    chi2, ok = np.array(chi2), np.array(ok)

    results[mod] = dict(tau=tau, tau_err=tau_err, chi2=chi2, ok=ok)

    flags = np.where(ok, '', '   no limite / invalido')
    for fm, t, te, c2, f in zip(FITMAX_SCAN, tau, tau_err, chi2, flags):
        print(f'  FIT_MAX = {fm:4.1f} us   tau_slow = {t:7.1f} +- {te:5.1f} ns'
              f'   chi2/ndof = {c2:5.2f}{f}')

    fig, ax = plt.subplots()

    ax.errorbar(FITMAX_SCAN[ok], tau[ok], yerr=tau_err[ok], fmt='o', ms=4,
                color=COLORS[mod], label='ajuste ok')

    if (~ok).any():
        ax.errorbar(FITMAX_SCAN[~ok], tau[~ok], yerr=tau_err[~ok], fmt='o', ms=4,
                    color='red', label='no limite / invalido')

    ax.axvline(3.0, color='gray', ls=':', lw=0.8, label='FIT_MAX padrao (3 us)')
    ax.set_xlabel('FIT_MAX [us]')
    ax.set_ylabel(r'$\tau_{slow}$ (E = 0) [ns]')
    ax.set_title(f'{mod} -- canais {MODULE_CONFIG[mod]["channels"]}')
    ax.legend()

    fig.savefig(f'{OUTDIR}tau_slow_vs_fitmax_{mod}.png', dpi=200)
    plt.close(fig)

# os 3 modulos juntos

fig, ax = plt.subplots()

for mod in MODULES:
    ok = results[mod]['ok']
    ax.errorbar(FITMAX_SCAN[ok], results[mod]['tau'][ok],
                yerr=results[mod]['tau_err'][ok],
                fmt=MARKERS[mod], ms=4, color=COLORS[mod], label=mod)
    if (~ok).any():
        ax.scatter(FITMAX_SCAN[~ok], results[mod]['tau'][~ok],
                   marker='x', color=COLORS[mod])

ax.axvline(3.0, color='gray', ls=':', lw=0.8, label='FIT_MAX padrao (3 us)')
ax.set_xlabel('FIT_MAX [us]')
ax.set_ylabel(r'$\tau_{slow}$ (E = 0) [ns]')
ax.set_title('M5, M6, M8 -- canais medios do modulo')
ax.legend()

fig.savefig(f'{OUTDIR}tau_slow_vs_fitmax_all.png', dpi=200)
plt.close(fig)


# --- estudo 1b: tau_slow(FIT_MIN), FIT_MAX fixo no valor de producao -------
# Mesma ideia do estudo 1, com os eixos trocados: agora FIT_MAX fica fixo no
# valor escolhido por arapuca (o resultado do estudo 1, ja em uso no
# fitting-bertolini-avg.py) e quem varre e FIT_MIN, um pouco pra frente e
# pra tras do valor padrao (0.25 us).
#
#   M8 -> FIT_MAX = 2.95 us   M5, M6 -> FIT_MAX = 2.56 us (M6 usa o mesmo
#   valor do M5 por enquanto -- ver nota no MODULE_CONFIG de
#   fitting-bertolini-avg.py)

FIT_MAX_BY_MODULE = {'M5': 2.56, 'M6': 2.56, 'M8': 2.95}

FITMIN_SCAN = np.arange(0.05, 0.46, 0.05)   # us -- em torno do FIT_MIN padrao (0.25)

results_fitmin = {}

for mod in MODULES:
    fit_max = FIT_MAX_BY_MODULE[mod]

    print(f'\n=== {mod}: varredura em FIT_MIN (FIT_MAX = {fit_max} us) ===')

    waveform, sigma, kernel = load_module_waveform(mod)
    model = make_model(kernel)

    tau, tau_err, chi2, ok = [], [], [], []
    seed = dict(INITIAL_GUESSES)

    for fit_min in FITMIN_SCAN:
        m, is_ok = fit_window(model, waveform, sigma, fit_min, fit_max, seed)

        n = int(((time_array > fit_min) & (time_array < fit_max)).sum())

        tau.append(m.values['tau_slow'] * 1000)      # ns
        tau_err.append(m.errors['tau_slow'] * 1000)
        chi2.append(m.fval / (n - m.nfit))
        ok.append(is_ok)

        if m.valid:
            seed = m.values.to_dict()

    tau, tau_err = np.array(tau), np.array(tau_err)
    chi2, ok = np.array(chi2), np.array(ok)

    results_fitmin[mod] = dict(tau=tau, tau_err=tau_err, chi2=chi2, ok=ok)

    flags = np.where(ok, '', '   no limite / invalido')
    for fm, t, te, c2, f in zip(FITMIN_SCAN, tau, tau_err, chi2, flags):
        print(f'  FIT_MIN = {fm:4.2f} us   tau_slow = {t:7.1f} +- {te:5.1f} ns'
              f'   chi2/ndof = {c2:5.2f}{f}')

    fig, ax = plt.subplots()

    ax.errorbar(FITMIN_SCAN[ok], tau[ok], yerr=tau_err[ok], fmt='o', ms=4,
                color=COLORS[mod], label='ajuste ok')

    if (~ok).any():
        ax.errorbar(FITMIN_SCAN[~ok], tau[~ok], yerr=tau_err[~ok], fmt='o', ms=4,
                    color='red', label='no limite / invalido')

    ax.axvline(0.25, color='gray', ls=':', lw=0.8, label='FIT_MIN padrao (0.25 us)')
    ax.set_xlabel('FIT_MIN [us]')
    ax.set_ylabel(r'$\tau_{slow}$ (E = 0) [ns]')
    ax.set_title(f'{mod} -- canais {MODULE_CONFIG[mod]["channels"]}, '
                 f'FIT_MAX = {fit_max} us')
    ax.legend()

    fig.savefig(f'{OUTDIR}tau_slow_vs_fitmin_{mod}.png', dpi=200)
    plt.close(fig)

# os 3 modulos juntos

fig, ax = plt.subplots()

for mod in MODULES:
    ok = results_fitmin[mod]['ok']
    ax.errorbar(FITMIN_SCAN[ok], results_fitmin[mod]['tau'][ok],
                yerr=results_fitmin[mod]['tau_err'][ok],
                fmt=MARKERS[mod], ms=4, color=COLORS[mod],
                label=f'{mod} (FIT_MAX = {FIT_MAX_BY_MODULE[mod]} us)')
    if (~ok).any():
        ax.scatter(FITMIN_SCAN[~ok], results_fitmin[mod]['tau'][~ok],
                   marker='x', color=COLORS[mod])

ax.axvline(0.25, color='gray', ls=':', lw=0.8, label='FIT_MIN padrao (0.25 us)')
ax.set_xlabel('FIT_MIN [us]')
ax.set_ylabel(r'$\tau_{slow}$ (E = 0) [ns]')
ax.set_title('M5, M6, M8 -- canais medios do modulo, FIT_MAX de producao')
ax.legend(fontsize=8)

fig.savefig(f'{OUTDIR}tau_slow_vs_fitmin_all.png', dpi=200)
plt.close(fig)


# --- estudo 2: M6, sensibilidade a POSICAO da janela (FIT_MIN x FIT_MAX) ---
# Nao so o tamanho da janela (estudo 1), mas onde ela comeca. Um grid 2D com
# tau_slow e I_slow = Aslow*tau_slow anotados numericamente em cada celula --
# mais objetivo que comparar curvas 1D a olho.

FITMIN_GRID = np.arange(0.15, 0.66, 0.10)   # us
FITMAX_GRID = np.arange(2.0, 4.51, 0.5)     # us

print(f'\n=== M6: grid FIT_MIN x FIT_MAX '
      f'({len(FITMIN_GRID)}x{len(FITMAX_GRID)} = '
      f'{len(FITMIN_GRID) * len(FITMAX_GRID)} ajustes) ===')

waveform_m6, sigma_m6, kernel_m6 = load_module_waveform('M6')
model_m6 = make_model(kernel_m6)

tau_grid = np.full((len(FITMIN_GRID), len(FITMAX_GRID)), np.nan)
islow_grid = np.full_like(tau_grid, np.nan)

seed = dict(INITIAL_GUESSES)

for i, fmin in enumerate(FITMIN_GRID):
    for j, fmax in enumerate(FITMAX_GRID):
        if fmax - fmin < 1.0:    # janela minima sensata
            continue

        m, is_ok = fit_window(model_m6, waveform_m6, sigma_m6, fmin, fmax, seed)

        if m.valid:
            seed = m.values.to_dict()

        tau_grid[i, j] = m.values['tau_slow'] * 1000       # ns
        islow_grid[i, j] = m.values['Aslow'] * m.values['tau_slow']

        print(f'  FIT_MIN = {fmin:.2f}  FIT_MAX = {fmax:.2f}   '
              f'tau_slow = {tau_grid[i, j]:7.1f} ns   '
              f'I_slow = {islow_grid[i, j]:.4f}'
              + ('' if is_ok else '   no limite / invalido'))

cmap = plt.get_cmap('viridis').copy()
cmap.set_bad('#dddddd')

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

dx = (FITMAX_GRID[1] - FITMAX_GRID[0]) / 2
dy = (FITMIN_GRID[1] - FITMIN_GRID[0]) / 2
extent = [FITMAX_GRID[0] - dx, FITMAX_GRID[-1] + dx,
          FITMIN_GRID[0] - dy, FITMIN_GRID[-1] + dy]

for ax, grid, title, fmt in [
    (axes[0], tau_grid, r'$\tau_{slow}$ [ns]', '{:.0f}'),
    (axes[1], islow_grid, r'$I_{slow} = A_s\,\tau_s$ [a.u.]', '{:.3f}'),
]:
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, origin='lower', aspect='auto', cmap=cmap,
                    extent=extent)

    for i, fmin in enumerate(FITMIN_GRID):
        for j, fmax in enumerate(FITMAX_GRID):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(fmax, fmin, fmt.format(v), ha='center', va='center',
                        fontsize=7, color='white')

    ax.set_xlabel('FIT_MAX [us]')
    ax.set_ylabel('FIT_MIN [us]')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

fig.suptitle('M6 (E = 0): sensibilidade de ' r'$\tau_{slow}$'
              ' e ' r'$I_{slow}$' ' a posicao da janela de ajuste')
fig.tight_layout()

fig.savefig(f'{OUTDIR}M6_fitwindow_sensitivity.png', dpi=200)
plt.close(fig)

print('\ndone ->', OUTDIR)
