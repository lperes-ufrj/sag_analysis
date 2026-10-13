"""tau_slow em E = 0 em funcao de FIT_MAX.

Estudo independente do loop principal de fitting-bertolini.py: usa so o run
de campo nulo e varre o limite superior da janela de ajuste. Aqui tau_slow
pode ir ate 2.5 us, ao contrario do teto de 1.6 us do ajuste padrao -- o
objetivo e ver se a janela esta truncando a cauda lenta.

make_causal_kernel() e model() sao copias literais de fitting-bertolini.py.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from iminuit import Minuit
from iminuit.cost import LeastSquares

# caminhos ancorados no proprio script: funciona de qualquer diretorio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # sag_analysis/fitting
REPO_DIR = os.path.dirname(SCRIPT_DIR)                    # sag_analysis


# configuracao

SAMPLE = 16  # ns

CHANNEL = 2080  # 2080 ou 2081 (M8)

# rodada de 09/set/2026 do M8 (mesma do fitting-bertolini-avg.py), no
# repositorio: coincidence/selected_waveforms/20260909_125014/
SCAN_M8 = '20260909_125014'

CHANNEL_CONFIG = {
    2080: dict(template='template_42156_M8_1.txt', scan=SCAN_M8),
    2081: dict(template='template_42156_M8_2.txt', scan=SCAN_M8),
}

CFG = CHANNEL_CONFIG[CHANNEL]

RUN_0 = 39510   # campo nulo

FIT_MIN = 0.25                              # us
FITMAX_SCAN = np.arange(2.0, 5.001, 0.1)    # us

TAU_FAST_FIXED = 0.006
T0_CENTER = 1.008
T0_HALF = 0.010

LIMITS = {
    't0':       (T0_CENTER - T0_HALF, T0_CENTER + T0_HALF),
    'Aslow':    (0, 5),
    'Afast':    (0, 100),
    'tau_slow': (0.7, 2.5),
    'tau_fast': (0.001, 0.05),
}

initial_guesses = {
    't0': T0_CENTER,
    'Aslow': 1.16,
    'Afast': 17.1,
    'tau_slow': 1.4,
    'tau_fast': TAU_FAST_FIXED,
}

output_folder = os.path.join(SCRIPT_DIR, f'plots_bertolini/{CHANNEL}')

os.makedirs(f'{output_folder}/parameters', exist_ok=True)


# kernel e modelo

time_array = np.arange(0, 1024 * SAMPLE, SAMPLE) / 1000  # us

template = np.loadtxt(
    REPO_DIR + f'/calib/output/data/templates/francesca_template/{CFG["template"]}'
)


def make_causal_kernel(template, n_baseline=5):
    """Resposta impulsional pronta para o forward-fold.

    Tres coisas: (i) subtrai a baseline medida no pre-pulso, (ii) leva a
    origem temporal do kernel para o indice 0, (iii) normaliza para area
    unitaria (sum * dt = 1).

    Por que (ii): o template bruto sobe em ~1 us. Se ele entra como kernel
    com esse atraso proprio, o atraso se soma ao t0 do fit -- contagem dupla.
    Com a subida no indice 0, todo o atraso fica por conta do t0.

    Por que (iii): sem area unitaria a escala de Aslow/Afast fica arbitraria
    (absorve a amplitude em ADC do template) e A*tau deixa de ser a integral
    fisica da componente."""

    k = np.nan_to_num(np.asarray(template, dtype=float), nan=0.0)

    rise = int(np.argmax(np.abs(np.diff(k))))

    if rise > n_baseline:
        k = k - np.median(k[:rise - n_baseline])

    k = np.roll(k, -rise)

    # o roll enrola as amostras de pre-pulso para o fim do array; numa
    # convolucao linear elas virariam uma copia espuria em atraso grande
    if rise > 0:
        k[-rise:] = 0.0

    dt = SAMPLE / 1000.0

    return k / (k.sum() * dt)


def model(t, t0, Aslow, Afast, tau_slow, tau_fast):
    """Dupla exponencial (luz emitida) convoluida com a resposta do detector.

    Convolucao causal: mode='full' truncado nas N primeiras amostras, com
    FIT_KERNEL comecando em t=0. Assim o modelo sobe exatamente em t0.

    (mode='same' assume a origem do kernel no indice N/2 = 512, ou seja
    ~8.2 us; com o kernel subindo no indice ~65 isso deslocava o modelo em
    -7.1 us e jogava a borda de subida para fora da janela de fit.)"""

    dt = time_array[1] - time_array[0]

    I = np.zeros_like(time_array)

    mask = time_array >= t0

    I[mask] = (
        Aslow * np.exp(-(time_array[mask] - t0) / tau_slow)
        + Afast * np.exp(-(time_array[mask] - t0) / tau_fast)
    )

    full_model = fftconvolve(I, FIT_KERNEL, mode='full')[:len(time_array)] * dt

    return np.interp(t, time_array, full_model)


FIT_KERNEL = make_causal_kernel(template)


# dados de campo nulo

file = (REPO_DIR + f'/coincidence/selected_waveforms/{CFG["scan"]}/'
        f'channel_{CHANNEL}_coincidence_scan_run_0{RUN_0}_{CFG["scan"]}.csv')

coin = pd.read_csv(file)

waveform = coin['mean'].to_numpy()
waveform = waveform / waveform.max()

noise_region = time_array > 8
sigma = np.std(waveform[noise_region] - np.mean(waveform[noise_region]))


# varredura em FIT_MAX

def at_limit(m, name):
    lo, hi = m.limits[name]
    return abs(m.values[name] - lo) < 1e-6 or abs(m.values[name] - hi) < 1e-6


tau, tau_err, chi2, ok = [], [], [], []

seed = dict(initial_guesses)

for fit_max in FITMAX_SCAN:

    mask = (time_array > FIT_MIN) & (time_array < fit_max)

    x = time_array[mask]
    y = waveform[mask]
    yerr = sigma * np.ones_like(y)

    m = Minuit(LeastSquares(x, y, yerr, model), **seed)

    for _name, _lim in LIMITS.items():
        m.limits[_name] = _lim

    m.fixed['tau_fast'] = True

    m.migrad()
    m.hesse()

    tau.append(m.values['tau_slow'])
    tau_err.append(m.errors['tau_slow'])
    chi2.append(m.fval / (len(x) - m.nfit))
    ok.append(bool(m.valid and not any(at_limit(m, p) for p in m.parameters
                                       if not m.fixed[p])))

    if m.valid:
        seed = m.values.to_dict()

tau = np.array(tau) * 1000        # ns
tau_err = np.array(tau_err) * 1000
chi2 = np.array(chi2)
ok = np.array(ok)

print(f"{'FIT_MAX':>8} {'tau_slow [ns]':>20} {'chi2/ndof':>11}")
for i, fm in enumerate(FITMAX_SCAN):
    flag = '' if ok[i] else '   no limite / invalido'
    print(f'{fm:8.2f} {tau[i]:12.1f} +- {tau_err[i]:5.1f} {chi2[i]:11.2f}{flag}')


# figura

fig, ax = plt.subplots()

ax.errorbar(FITMAX_SCAN[ok], tau[ok], yerr=tau_err[ok], fmt='o', ms=4,
            label='ajuste ok')

if (~ok).any():
    ax.errorbar(FITMAX_SCAN[~ok], tau[~ok], yerr=tau_err[~ok], fmt='o', ms=4,
                color='red', label='no limite / invalido')

ax.set_xlabel('FIT_MAX [us]')
ax.set_ylabel(r'$\tau_{slow}$ (E = 0) [ns]')
ax.set_title(f'canal {CHANNEL}')
ax.legend()

fig.savefig(f'{output_folder}/parameters/tau_slow_vs_fitmax.png', dpi=300)
plt.close(fig)
