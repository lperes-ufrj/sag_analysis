"""Fit forward-fold da cintilacao do LAr dopado com Xe, com transferencia
de energia Ar -> Xe.

Por que nao a dupla exponencial do fitting-bertolini-avg.py: com Xe o pulso
sobe devagar ate um maximo (~1.9 us em 10 ppm no M8) antes de cair. Isso e a
luz do Xe aparecendo com atraso -- o tripleto do Ar2* transfere energia para
o Xe e a emissao do Xe2* (175 nm) so vem depois de uma etapa intermediaria
(ArXe*). A dupla exponencial nao tem termo de subida e nao descreve isso.

Cinetica (por run, sem impor dependencia em [Xe] no fit):

    Ar2*(singleto)  --> 128 nm, tau_f (fixo, nao resolvido a 16 ns)
    Ar2*(tripleto)  --> 128 nm, tau_d = 1/(1/tau_T + k_T [Xe])
         |__ k_T [Xe] --> ArXe* --(tau_r)--> Xe2* --> 175 nm (rapido)

Luz emitida, para t > t0 (dt = t - t0):

    I(t) = A_f e^{-dt/tau_f}                                  singleto
         + A_s e^{-dt/tau_d}                                  tripleto do Ar
         + I_x/(tau_d - tau_r) (e^{-dt/tau_d} - e^{-dt/tau_r})  luz do Xe

O termo do Xe decai com o tripleto (de onde vem a energia) e sobe com tau_r;
I_x e a sua integral. As integrais sao I_f = A_f tau_f, I_s = A_s tau_d e I_x.
tau_r = f_r * tau_d e o tempo de vida do intermediario (ArXe*). Ele pode ser
mais curto OU mais longo que tau_d (f_r < 1 ou > 1): o termo do Xe e simetrico
em (tau_d, tau_r) e, para tau_r -> tau_d, tende a I_x t/tau^2 e^{-t/tau}. O que
quebra a simetria e o tripleto residual do Ar, que decai sempre com tau_d.

A emissao e integrada em cada bin de 16 ns (e nao amostrada no centro, como
no -avg), entao A*tau e a integral fisica mesmo para o singleto de 6 ns.

Duas etapas:
  1. fit livre por run (diagnostico). Em baixa [Xe] o tripleto do Ar e o
     termo do Xe podem trocar de papel, entao os tempos daqui nao sao o
     resultado. No run de 0.01 ppm, I_x = 0 (LAr praticamente puro).
  2. fit global de todos os runs, com os tempos ligados a [Xe]:
        1/tau_d = 1/tau_T + k_d [Xe]     (tripleto depletado pelo Xe)
        1/tau_r = a_r     + k_r [Xe]     (vida do intermediario ArXe*)
     e, por run, so t0, A_f, A_s e I_x livres. E o resultado.
Saem tambem as integrais de cada componente e a luz total relativa ao run
de 0.01 ppm.

Dados: rodadas do Xe do Leo (01/out/2026), 7 runs de 0.01 a 10 ppm, media
dos dois canais do modulo, normalizada pelo pico do run de 0.01 ppm.

    python fitting-xe-transfer.py [M8|M6|M5]

Saidas em plots_xe_transfer/<modulo>/.
"""
import datetime as _dt
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dunestyle.matplotlib as dunestyle  # noqa: F401  (estilo DUNE no import)
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.signal import fftconvolve

# caminhos ancorados no proprio script: funciona de qualquer diretorio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # sag_analysis/fitting
REPO_DIR = os.path.dirname(SCRIPT_DIR)                    # sag_analysis

SAMPLE = 16                                               # ns
DT = SAMPLE / 1000.0                                      # us
time_array = np.arange(0, 1024) * DT                      # us

# =============================================================================
# CONFIGURACAO
# =============================================================================

# rodadas de coincidencia do Xe (mesma selecao das rodadas de 09/set)
SCAN_M8 = '20261001_142744'    # M7 + M8
SCAN_LOW = '20261001_140103'   # M5 + M6
PATTERN = 'channel_{ch}_coincidence_scan_run_0{run}_{scan}.csv'

MODULE_CONFIG = {
    'M8': dict(channels=(2080, 2081), scan=SCAN_M8,
               templates=('template_42156_M8_1.txt', 'template_42156_M8_2.txt')),
    'M6': dict(channels=(2060, 2061), scan=SCAN_LOW,
               templates=('template_40808_M6_1.txt', 'template_40808_M6_2.txt')),
    'M5': dict(channels=(2050, 2051), scan=SCAN_LOW,
               templates=('template_41522_M5_1.txt', 'template_41522_M5_2.txt')),
}
MODULE_DEFAULT = 'M8'

# [Xe] em ppm -> run (analysis/LY_vs_Xeppm/ e sufixo _<x>ppm dos ROOT em
# coincidence/input_lists/xe_input_lists/). Ordem crescente: o fit de cada
# run parte do resultado do anterior.
runs = {0.01: 43440, 1: 43552, 2: 43717, 3: 43790, 5: 43903, 7: 44010, 10: 44108}
X_REF = 0.01                   # referencia: normalizacao e razoes

TAU_FAST = 0.006               # us, singleto do LAr (fixo)
T0_REF = 1.008                 # us, t0 do ajuste do -avg em E = 0 (M8)

# janela do fit. O teto e maior que no -avg (2.95/2.56 us): com Xe o maximo
# vai para ~2 us e a queda precisa de alcance para medir tau_d. Ainda bem
# antes de t > 8 us, onde esta o undershoot.
FIT_MIN = 0.25                 # us
FIT_MAX = 4.0                  # us

NOISE_REGION_T = 8.0           # us, yerr = std do dado depois disso

LIMITS = {
    't0':    (0.5, 1.3),
    'A_f':   (0, 300),
    'A_s':   (0, 50),
    'I_x':   (0, 100),
    'tau_d': (0.1, 3.0),
    'f_r':   (0.005, 5.0),
}

SEED0 = dict(t0=T0_REF, A_f=20.0, A_s=1.0, I_x=0.1, tau_d=1.4, f_r=0.3)


def escolhe_modulo():
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip().upper()
        if arg not in MODULE_CONFIG:
            raise SystemExit(f'modulo invalido: {sys.argv[1]!r}. '
                             f'Validos: {", ".join(MODULE_CONFIG)}')
        return arg
    if not sys.stdin.isatty():
        return MODULE_DEFAULT
    resp = input(f'Modulo ({"/".join(MODULE_CONFIG)}) '
                 f'[enter = {MODULE_DEFAULT}]: ').strip().upper()
    return resp if resp in MODULE_CONFIG else MODULE_DEFAULT


MODULE = escolhe_modulo()
CFG = MODULE_CONFIG[MODULE]
print(f'\n>>> modulo {MODULE}  (canais {CFG["channels"][0]} + {CFG["channels"][1]})\n')

output_folder = os.path.join(SCRIPT_DIR, f'plots_xe_transfer/{MODULE}')
for sub in ('fit', 'parameters'):
    os.makedirs(f'{output_folder}/{sub}', exist_ok=True)

# =============================================================================
# DADOS E KERNEL
# =============================================================================


def coincidence_file(run, ch):
    return (f'{REPO_DIR}/coincidence/selected_waveforms/{CFG["scan"]}/'
            + PATTERN.format(ch=ch, run=run, scan=CFG['scan']))


def coincidence_mean(run):
    """Media dos dois canais do modulo."""
    dfs = [pd.read_csv(coincidence_file(run, ch)) for ch in CFG['channels']]
    return np.mean([df['mean'].to_numpy() for df in dfs], axis=0)


def make_causal_kernel(template, n_baseline=5):
    """Igual ao do fitting-bertolini-avg.py: baseline do pre-pulso subtraida,
    subida levada ao indice 0 e area unitaria (sum * dt = 1)."""
    k = np.nan_to_num(np.asarray(template, dtype=float), nan=0.0)
    rise = int(np.argmax(np.abs(np.diff(k))))
    if rise > n_baseline:
        k = k - np.median(k[:rise - n_baseline])
    k = np.roll(k, -rise)
    if rise > 0:
        k[-rise:] = 0.0
    return k / (k.sum() * DT)


templates = [np.loadtxt(f'{REPO_DIR}/calib/output/data/templates/'
                        f'francesca_template/{t}') for t in CFG['templates']]
FIT_KERNEL = 0.5 * (make_causal_kernel(templates[0])
                    + make_causal_kernel(templates[1]))

# =============================================================================
# MODELO
# =============================================================================

_edges_lo = time_array
_edges_hi = time_array + DT


def _exp_bins(t0, tau):
    """Taxa media de e^{-(t-t0)/tau} theta(t-t0) em cada bin [t_i, t_i+dt].
    sum(...) * dt = tau exatamente (a menos da janela finita)."""
    a = np.maximum(_edges_lo, t0)
    b = np.maximum(_edges_hi, t0)
    return tau * (np.exp(-(a - t0) / tau) - np.exp(-(b - t0) / tau)) / DT


def components(t0, A_f, A_s, I_x, tau_d, f_r):
    """Luz emitida por componente, por bin (antes da convolucao)."""
    if abs(f_r - 1.0) < 1e-6:          # limite tau_r = tau_d (evita 0/0)
        f_r = 1.0 + 1e-6
    tau_r = f_r * tau_d
    e_d = _exp_bins(t0, tau_d)
    e_r = _exp_bins(t0, tau_r)
    return {
        'singleto':      A_f * _exp_bins(t0, TAU_FAST),
        'tripleto Ar':   A_s * e_d,
        'Xe (subida)':   I_x / (tau_d - tau_r) * (e_d - e_r),
    }


def fold(light):
    return fftconvolve(light, FIT_KERNEL, mode='full')[:len(time_array)] * DT


def model(t, t0, A_f, A_s, I_x, tau_d, f_r):
    light = sum(components(t0, A_f, A_s, I_x, tau_d, f_r).values())
    return np.interp(t, time_array, fold(light))


def t_cross(wf, frac=0.1):
    """Primeiro instante em que wf passa de frac * max (borda de subida)."""
    i = int(np.argmax(wf > frac * np.nanmax(wf)))
    return time_array[i]

# =============================================================================
# ETAPA 1 -- FIT LIVRE POR RUN (diagnostico)
# =============================================================================
# Cada run com todos os parametros livres. Serve para ver o que o dado pede,
# mas o modelo tem degenerescencias (o tripleto do Ar e o termo do Xe podem
# trocar de papel em baixa [Xe]), entao os tempos daqui NAO sao o resultado.
# No run de referencia (0.01 ppm) I_x = 0 fixo: LAr praticamente puro.

waveforms = {x: coincidence_mean(r) for x, r in runs.items()}
norm = np.nanmax(waveforms[X_REF])
waveforms = {x: w / norm for x, w in waveforms.items()}

mask = (time_array > FIT_MIN) & (time_array < FIT_MAX)
xs = time_array[mask]


def noise_sigma(wf):
    noise = wf[time_array > NOISE_REGION_T]
    return np.std(noise - np.mean(noise))


t10_ref = t_cross(waveforms[X_REF])
sigmas = {x: noise_sigma(wf) for x, wf in waveforms.items()}

free, seed = {}, dict(SEED0)
for x, run in runs.items():
    wf = waveforms[x]
    ys = wf[mask]
    # t0 semeado pela borda de subida do proprio run, relativa a referencia
    seed['t0'] = T0_REF + (t_cross(wf) - t10_ref)
    if x == X_REF:
        seed['I_x'] = 0.0

    m = Minuit(LeastSquares(xs, ys, sigmas[x] * np.ones_like(ys), model), **seed)
    for name, lim in LIMITS.items():
        m.limits[name] = lim
    if x == X_REF:
        m.fixed['I_x'] = True
        m.fixed['f_r'] = True
    m.migrad(ncall=20000)
    if not m.valid:
        m.simplex().migrad(ncall=20000)
    m.hesse()
    free[x] = m
    if m.valid:
        seed = m.values.to_dict()
        if x == X_REF:
            seed['I_x'] = 0.5
    print(f'[etapa 1] {x:>5} ppm  chi2/ndof = {m.fval / (len(xs) - m.nfit):6.2f}  '
          f'tau_d = {m.values["tau_d"]:.3f}  tau_r = {m.values["f_r"] * m.values["tau_d"]:.3f}  '
          f'I_x = {m.values["I_x"]:.3f}  valid = {m.valid}')

# =============================================================================
# ETAPA 2 -- FIT GLOBAL COM A CINETICA IMPOSTA (resultado)
# =============================================================================
# Todos os runs ao mesmo tempo, com os tempos ligados a [Xe]:
#     1/tau_d(x) = 1/tau_T + k_d x      (tripleto do Ar depletado pelo Xe)
#     1/tau_r(x) = a_r     + k_r x      (vida do intermediario ArXe*)
# Por run ficam livres so t0, A_f, A_s e I_x (I_x = 0 no run de referencia).
# Isso tira a troca de papeis da etapa 1: os tempos saem de todos os runs.

X = list(runs)
GLOBAL_NAMES = ['tau_T', 'k_d', 'a_r', 'k_r']
PER_RUN = ['t0', 'A_f', 'A_s', 'I_x']


def pname(p, i):
    return f'{p}_{i}'


names = GLOBAL_NAMES + [pname(p, i) for i in range(len(X)) for p in PER_RUN]


def taus(x, tau_T, k_d, a_r, k_r):
    tau_d = 1.0 / (1.0 / tau_T + k_d * x)
    tau_r = 1.0 / (a_r + k_r * x)
    return tau_d, tau_r


def model_global_run(i, par):
    tau_T, k_d, a_r, k_r = par[:4]
    t0, A_f, A_s, I_x = par[4 + 4 * i: 8 + 4 * i]
    tau_d, tau_r = taus(X[i], tau_T, k_d, a_r, k_r)
    return model(xs, t0, A_f, A_s, I_x, tau_d, tau_r / tau_d)


ys_all = [waveforms[x][mask] for x in X]
ws_all = [1.0 / sigmas[x]**2 for x in X]


def chi2_global(*par):
    return sum(np.sum(w * (y - model_global_run(i, par))**2)
               for i, (y, w) in enumerate(zip(ys_all, ws_all)))


chi2_global.errordef = Minuit.LEAST_SQUARES

p0 = [free[X_REF].values['tau_d'], 0.2, 0.3, 0.2]
for x in X:
    v = free[x].values
    p0 += [v['t0'], v['A_f'], v['A_s'], 0.0 if x == X_REF else max(v['I_x'], 0.1)]

mg = Minuit(chi2_global, *p0, name=names)
mg.limits['tau_T'] = (0.3, 5.0)
mg.limits['k_d'] = (0, 5)
mg.limits['a_r'] = (0, 50)
mg.limits['k_r'] = (0, 50)
for i in range(len(X)):
    mg.limits[pname('t0', i)] = LIMITS['t0']
    mg.limits[pname('A_f', i)] = LIMITS['A_f']
    mg.limits[pname('A_s', i)] = LIMITS['A_s']
    mg.limits[pname('I_x', i)] = LIMITS['I_x']
mg.fixed[pname('I_x', X.index(X_REF))] = True

mg.migrad(ncall=200000)
if not mg.valid:
    mg.simplex(ncall=200000).migrad(ncall=200000)
mg.hesse()

ndof_g = len(X) * len(xs) - mg.nfit
print('\n=== etapa 2: fit global ===')
for n in GLOBAL_NAMES:
    print(f'{n:6s} = {mg.values[n]:.5g} +- {mg.errors[n]:.3g}')
print(f'chi2/ndof = {mg.fval:.1f}/{ndof_g} = {mg.fval / ndof_g:.2f}  valid = {mg.valid}')

G = {n: mg.values[n] for n in GLOBAL_NAMES}
tau_d_g, tau_r_g = taus(np.array(X, dtype=float), **G)

# chi2 por run no fit global
chi2_run = [np.sum(ws_all[i] * (ys_all[i] - model_global_run(i, mg.values))**2)
            for i in range(len(X))]

# plots por run (fit global), com as componentes
for i, x in enumerate(X):
    wf = waveforms[x]
    t0, A_f, A_s, I_x = [mg.values[pname(p, i)] for p in PER_RUN]
    f_r = tau_r_g[i] / tau_d_g[i]
    comps = components(t0, A_f, A_s, I_x, tau_d_g[i], f_r)
    ndof_i = len(xs) - 4
    for yscale in ('log', 'linear'):
        fig, ax = plt.subplots()
        ax.scatter(time_array, wf, s=3, color='k', label='dado')
        ax.plot(xs, model_global_run(i, mg.values), 'r-', lw=1.2,
                label=rf'fit global, $\chi^2$/ndof = {chi2_run[i] / ndof_i:.2f}')
        for (name, c), col in zip(comps.items(), ('C0', 'C2', 'C1')):
            ax.plot(time_array, fold(c), '--', lw=1, color=col, label=name)
        ax.axvspan(FIT_MIN, FIT_MAX, color='gray', alpha=0.15)
        ax.set_xlim(0.25, 6)
        if yscale == 'log':
            ax.set_yscale('log')
            ax.set_ylim(3e-3, 1.5 * np.nanmax(wf))
        ax.set_xlabel('time [us]')
        ax.set_ylabel('amplitude / pico(0.01 ppm)')
        ax.set_title(f'{MODULE}, {x} ppm   '
                     rf'$\tau_d$={tau_d_g[i]:.3f} us, $\tau_r$={tau_r_g[i]:.3f} us',
                     fontsize=10)
        ax.legend(fontsize=7)
        fig.savefig(f'{output_folder}/fit/fit_{x}ppm_{yscale}.png', dpi=200)
        plt.close(fig)

fig, ax = plt.subplots()
for x, wf in waveforms.items():
    ax.plot(time_array, wf, lw=1, label=f'{x} ppm')
ax.set_xlim(0.25, 6)
ax.set_xlabel('time [us]')
ax.set_ylabel('amplitude / pico(0.01 ppm)')
ax.legend(fontsize=7)
fig.savefig(f'{output_folder}/fit/waveforms_all_ppm.png', dpi=200)
plt.close(fig)

# =============================================================================
# DERIVADOS
# =============================================================================

xe = np.array(X, dtype=float)
iref = X.index(X_REF)


def gv(p):
    return np.array([mg.values[pname(p, i)] for i in range(len(X))])


def ge(p):
    return np.array([mg.errors[pname(p, i)] for i in range(len(X))])


def gcov(a, b):
    return mg.covariance[a, b]


# erros de tau_d(x), tau_r(x) pela covariancia dos parametros globais
def tau_err(x, which):
    eps = 1e-6
    names_ = ('tau_T', 'k_d') if which == 'd' else ('a_r', 'k_r')
    grad = []
    for n in names_:
        g1, g2 = dict(G), dict(G)
        g1[n] += eps
        g2[n] -= eps
        idx = 0 if which == 'd' else 1
        grad.append((taus(x, **g1)[idx] - taus(x, **g2)[idx]) / (2 * eps))
    c = np.array([[gcov(a, b) for b in names_] for a in names_])
    grad = np.array(grad)
    return np.sqrt(np.einsum('i...,ij,j...->...', grad, c, grad))


tau_d_e = tau_err(xe, 'd')
tau_r_e = tau_err(xe, 'r')

I_f, I_f_err = gv('A_f') * TAU_FAST, ge('A_f') * TAU_FAST
I_s = gv('A_s') * tau_d_g
I_s_err = np.sqrt((ge('A_s') * tau_d_g)**2 + (gv('A_s') * tau_d_e)**2)
I_x, I_x_err = gv('I_x'), ge('I_x')
I_tot = I_f + I_s + I_x
I_tot_err = np.sqrt(I_f_err**2 + I_s_err**2 + I_x_err**2)   # sem covariancias
LY_rel = I_tot / I_tot[iref]
LY_rel_err = LY_rel * np.sqrt((I_tot_err / I_tot)**2 + (I_tot_err[iref] / I_tot[iref])**2)
LY_rel_err[iref] = 0.0
f_xe = I_x / I_tot

# etapa 1 para comparacao
tau_d_1 = np.array([free[x].values['tau_d'] for x in X])
tau_d_1e = np.array([free[x].errors['tau_d'] for x in X])
tau_r_1 = np.array([free[x].values['f_r'] * free[x].values['tau_d'] for x in X])

# =============================================================================
# PLOTS VS [Xe]
# =============================================================================

xx = np.linspace(0, xe.max() * 1.05, 300)
txd, txr = taus(xx, **G)

fig, ax = plt.subplots()
ax.plot(xx, txd, 'C0-', lw=1.2, label=(rf'$\tau_d$: $\tau_T$ = {G["tau_T"]:.3f} us, '
                                       rf'$k_d$ = {G["k_d"]:.3f} us$^{{-1}}$ppm$^{{-1}}$'))
ax.plot(xx, txr, 'C1-', lw=1.2, label=(rf'$\tau_r$: $1/a_r$ = {1 / G["a_r"] if G["a_r"] > 0 else np.inf:.3g} us, '
                                       rf'$k_r$ = {G["k_r"]:.3f} us$^{{-1}}$ppm$^{{-1}}$'))
ax.plot(xe, tau_d_1, 'o', mfc='none', color='C0', label=r'$\tau_d$ etapa 1 (livre)')
ax.plot(xe[xe != X_REF], tau_r_1[xe != X_REF], 's', mfc='none', color='C1',
        label=r'$\tau_r$ etapa 1 (livre)')
ax.set_yscale('log')
ax.set_xlabel('Xe concentration [ppm]')
ax.set_ylabel('tempo [us]')
ax.legend(fontsize=7)
fig.savefig(f'{output_folder}/parameters/taus_vs_xe.png', dpi=200)
plt.close(fig)

fig, ax = plt.subplots()
for y, e, lab in ((I_f, I_f_err, r'$I_f$ (singleto)'), (I_s, I_s_err, r'$I_s$ (tripleto Ar)'),
                  (I_x, I_x_err, r'$I_x$ (Xe)'), (I_tot, I_tot_err, 'total')):
    ax.errorbar(xe, y, yerr=e, fmt='o-', ms=4, lw=0.8, label=lab)
ax.set_xlabel('Xe concentration [ppm]')
ax.set_ylabel('integral / pico(0.01 ppm) [us]')
ax.legend(fontsize=8)
fig.savefig(f'{output_folder}/parameters/integrals.png', dpi=200)
plt.close(fig)

fig, ax = plt.subplots()
ax.errorbar(xe, LY_rel, yerr=LY_rel_err, fmt='o', ms=4)
ax.set_xlabel('Xe concentration [ppm]')
ax.set_ylabel('luz total / (0.01 ppm)')
fig.savefig(f'{output_folder}/parameters/light_relative.png', dpi=200)
plt.close(fig)

fig, ax = plt.subplots()
ax.errorbar(xe, gv('t0'), yerr=ge('t0'), fmt='o', ms=4)
ax.set_xlabel('Xe concentration [ppm]')
ax.set_ylabel(r'$t_0$ [us]')
fig.savefig(f'{output_folder}/parameters/t0.png', dpi=200)
plt.close(fig)

# =============================================================================
# TABELAS
# =============================================================================


def _write_table(path, meta, columns):
    names_ = [c[0] for c in columns]
    cols = [list(c[1]) for c in columns]
    fmt = lambda v: ('1' if v else '0') if isinstance(v, (bool, np.bool_)) else (
        str(int(v)) if isinstance(v, (int, np.integer)) else f'{float(v):.8g}')
    body = [[fmt(c[i]) for c in cols] for i in range(len(cols[0]))]
    w = [max([len(n)] + [len(r[j]) for r in body]) for j, n in enumerate(names_)]
    with open(path, 'w') as fh:
        fh.write('  '.join(n.rjust(w[j]) for j, n in enumerate(names_)) + '\n')
        for line in meta:
            fh.write(f'# {line}\n')
        fh.write('#\n')
        for r in body:
            fh.write('  '.join(r[j].rjust(w[j]) for j in range(len(names_))) + '\n')
    print(f'[txt] {path}')


meta = [
    f'gerado em     : {_dt.datetime.now():%Y-%m-%d %H:%M:%S}',
    f'script        : {os.path.basename(__file__)}',
    f'modulo        : {MODULE}  canais {CFG["channels"]}',
    f'template      : {" + ".join(CFG["templates"])}',
    f'dados         : coincidence/selected_waveforms/{CFG["scan"]}',
    f'janela do fit : {FIT_MIN} - {FIT_MAX} us;  tau_f fixo em {TAU_FAST} us',
    'modelo        : A_f e^-t/tau_f + A_s e^-t/tau_d + I_x/(tau_d-tau_r)(e^-t/tau_d - e^-t/tau_r)',
    'fit global    : 1/tau_d = 1/tau_T + k_d [Xe];  1/tau_r = a_r + k_r [Xe]',
    *[f'{n:6s}        : {mg.values[n]:.6g} +- {mg.errors[n]:.3g}' for n in GLOBAL_NAMES],
    f'chi2/ndof     : {mg.fval:.1f}/{ndof_g} = {mg.fval / ndof_g:.3f}  valid={mg.valid}',
    'unidades      : [Xe] ppm; tempos us; taxas 1/us e 1/(us ppm); amplitudes/integrais '
    'normalizadas pelo pico de 0.01 ppm',
]

_write_table(f'{output_folder}/parameters/fit_global_vs_xe.txt', meta, [
    ('xe_ppm', X), ('run', [runs[x] for x in X]),
    ('t0', gv('t0')), ('t0_err', ge('t0')),
    ('A_f', gv('A_f')), ('A_f_err', ge('A_f')),
    ('A_s', gv('A_s')), ('A_s_err', ge('A_s')),
    ('I_x', I_x), ('I_x_err', I_x_err),
    ('tau_d', tau_d_g), ('tau_d_err', tau_d_e),
    ('tau_r', tau_r_g), ('tau_r_err', tau_r_e),
    ('I_f', I_f), ('I_f_err', I_f_err), ('I_s', I_s), ('I_s_err', I_s_err),
    ('I_total', I_tot), ('I_total_err', I_tot_err),
    ('LY_rel', LY_rel), ('LY_rel_err', LY_rel_err), ('f_xe', f_xe),
    ('chi2_run', chi2_run), ('ndof_run', [len(xs) - 4] * len(X)),
])

_write_table(f'{output_folder}/parameters/fit_free_vs_xe.txt', meta[:7] + [
    'etapa 1: fit livre por run (diagnostico; tempos podem trocar de papel)',
], [('xe_ppm', X)] + [
    (c, [getattr(free[x], 'values' if not c.endswith('_err') else 'errors')[c.replace('_err', '')]
         for x in X])
    for n in ('t0', 'A_f', 'A_s', 'I_x', 'tau_d', 'f_r') for c in (n, f'{n}_err')
] + [('chi2_ndof', [free[x].fval / (len(xs) - free[x].nfit) for x in X]),
     ('valid', [free[x].valid for x in X])])
