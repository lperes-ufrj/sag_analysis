import os

import matplotlib.pyplot as plt
import scipy as sci
import pandas as pd
import numpy as np
from scipy.special import erfc, erfcx
import dunestyle.matplotlib as dunestyle
from iminuit import Minuit
from iminuit.cost import LeastSquares

# caminhos ancorados no proprio script: funciona de qualquer diretorio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # sag_analysis/fitting
REPO_DIR = os.path.dirname(SCRIPT_DIR)                    # sag_analysis

SAMPLE = 16  # ns

TIME_WINDOW = 1024 * SAMPLE / 1000  # us

NYQUIST_FREQUENCY = 1 / (2 * SAMPLE / 1000)   # = 31.25 → MHz

FREQUENCY_WINDOW = 1 / TIME_WINDOW

# canal (array de SiPM) analisado:
#   2080/2081 (M8), 2060/2061 (M6), 2050/2051 (M5).
CHANNEL = 2061

# por canal: template da Francesca e rodada de coincidencia
# (subpasta de coincidence/selected_waveforms/, a mesma publicada no GitHub
# lperes-ufrj/sag_analysis). Rodada de 09/set/2026, igual ao
# fitting-bertolini-avg.py:
#   20260909_125014 : M7 + M8, gatilho 2030-2031-2040-2041 vs 2050-2051-2060-2061
#   20260909_213507 : M5 + M6, gatilho 2070-2071-2080-2081 vs 2010-2011-2020-2021
# Nas duas ha um limiar de amplitude por canal/run (taxas equalizadas) antes
# da coincidencia -- NAO misturar com as rodadas antigas.
SCAN_M8 = '20260909_125014'
SCAN_LOW = '20260909_213507'

PATTERN = 'channel_{ch}_coincidence_scan_run_0{run}_{scan}.csv'

CHANNEL_CONFIG = {
    2080: dict(template='template_42156_M8_1.txt', scan=SCAN_M8,  skip_fields=()),
    2081: dict(template='template_42156_M8_2.txt', scan=SCAN_M8,  skip_fields=()),
    2060: dict(template='template_40808_M6_1.txt', scan=SCAN_LOW, skip_fields=()),
    2061: dict(template='template_40808_M6_2.txt', scan=SCAN_LOW, skip_fields=()),
    2050: dict(template='template_41522_M5_1.txt', scan=SCAN_LOW, skip_fields=()),
    2051: dict(template='template_41522_M5_2.txt', scan=SCAN_LOW, skip_fields=()),
}

CFG = CHANNEL_CONFIG[CHANNEL]


def coincidence_file(run):
    return (REPO_DIR + f'/coincidence/selected_waveforms/{CFG["scan"]}/'
            + PATTERN.format(ch=CHANNEL, run=run, scan=CFG['scan']))


output_folder = os.path.join(SCRIPT_DIR, f'plots/{CHANNEL}')

for folder in ['templates', 'fft', 'deconv', 'fits', 'fit', 'parameters']:
    os.makedirs(f'{output_folder}/{folder}', exist_ok=True)

def gaussian(x, t0, sigma):
    return np.exp(-((x - t0)**2) / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

def double_exp(x, t0, Aslow, Afast, tau_slow, tau_fast):
    dt = x - t0
    return np.where(dt > 0,
                    Aslow*np.exp(-dt/tau_slow) + Afast*np.exp(-dt/tau_fast))

def _emg(t, A, t0, tau, sigma):
    """Exponencial unilateral A*exp(-(t-t0)/tau)*theta(t-t0) convoluida
    com gaussiana normalizada de largura sigma. Integral exata = A*tau."""
    dt = t - t0
    z = sigma / (tau * np.sqrt(2)) - dt / (sigma * np.sqrt(2))

    # dois ramos: erfcx estavel para z>0, forma direta estavel para z<0
    out = np.empty_like(z, dtype=float)

    p = z >= 0
    out[p] = np.exp(-dt[p]**2 / (2 * sigma**2)) * erfcx(z[p])

    n = ~p
    out[n] = np.exp(sigma**2 / (2 * tau**2) - dt[n] / tau) * erfc(z[n])

    return 0.5 * A * out

def model(t, t0, Aslow, Afast, tau_slow, tau_fast, sigma):
    return (_emg(t, Aslow, t0, tau_slow, sigma)
            + _emg(t, Afast, t0, tau_fast, sigma)
            )


# templates francesca

time_array = np.arange(0, 1024 * SAMPLE, SAMPLE) / 1000  # us

frequency_array = np.fft.rfftfreq(len(time_array), SAMPLE / 1000)  # MHz

template_m8_1 = np.loadtxt(
    REPO_DIR + '/calib/output/data/templates/francesca_template/template_42156_M8_1.txt'
)

template_m8_2 = np.loadtxt(
    REPO_DIR + '/calib/output/data/templates/francesca_template/template_42156_M8_2.txt'
)

plt.figure()
plt.scatter(time_array, template_m8_1, s=3, label='2080')
plt.scatter(time_array, template_m8_2, s=3, label='2081')
plt.xlabel('time [us]')
plt.ylabel('amplitude [ADC]')
plt.legend()
plt.savefig(f'{output_folder}/templates/templates_francesca.png', dpi=300)
plt.close()

# my templates

template_2080 = np.load(
    REPO_DIR + '/calib/output/data/templates/041256/template_2080.npy'
)

template_2081 = np.load(
    REPO_DIR + '/calib/output/data/templates/041256/template_2081.npy'
)

plt.figure()
plt.scatter(time_array, template_2080, s=3, label='2080')
plt.scatter(time_array, template_2081, s=3, label='2081')
plt.xlabel('time [us]')
plt.ylabel('amplitude [ADC]')
plt.legend()
plt.savefig(f'{output_folder}/templates/templates_mine.png', dpi=300)
plt.close()

# pareamento canal <-> template

TEMPLATE = np.loadtxt(
    REPO_DIR + f'/calib/output/data/templates/francesca_template/{CFG["template"]}'
)

# templates proprios so foram feitos para 2080/2081; sem eles os plots de
# diagnostico *_mine repetem o da Francesca
TEMPLATE_MINE = {2080: template_2080, 2081: template_2081}.get(CHANNEL, TEMPLATE)

if CHANNEL not in (2080, 2081):
    print(f'[aviso] canal {CHANNEL} nao tem template proprio: '
          'os plots *_mine repetem o template da Francesca')

# nomes usados no restante do script
template_m8_2 = TEMPLATE
template_2081 = TEMPLATE_MINE

def shift_no_wrap(arr, shift):
    """Like np.roll, but pads with NaN instead of wrapping the samples
    that fall off one end around to the other -- avoids the fake step
    discontinuity a circular roll leaves at the wrap seam."""
    shifted = np.full_like(arr, np.nan, dtype=float)
    if shift > 0:
        shifted[shift:] = arr[:-shift]
    elif shift < 0:
        shifted[:shift] = arr[-shift:]
    else:
        shifted[:] = arr
    return shifted


def deconvolve_align(coin_mean, template, noalign_path, align_path):

    # no alignment

    plt.figure()

    plt.scatter(
        time_array,
        coin_mean / np.max(coin_mean),
        s=3,
        label='Coincidence'
    )
    plt.scatter(
        time_array,
        template / np.max(template),
        s=3,
        label='Template'
    )
    plt.xlabel('time [us]')
    plt.ylabel('amplitude [adu]')
    plt.legend()
    plt.savefig(noalign_path, dpi=300)
    plt.close()

    # correlation

    correlation = np.correlate(
        coin_mean - np.median(coin_mean),
        template - np.median(template),
        mode='full'
    )

    shift = np.argmax(correlation) - (len(template) - 1)

    # align template (no-wrap: avoids a fake step at the shift seam)

    template_align = shift_no_wrap(template, shift)

    # aligned plot

    plt.figure()

    plt.scatter(
        time_array,
        coin_mean / np.max(coin_mean),
        s=3,
        label='Coincidence'
    )
    plt.scatter(
        time_array,
        template_align / np.nanmax(template_align),
        s=3,
        label='Template'
    )
    plt.xlabel('time [us]')
    plt.ylabel('amplitude [adu]')
    plt.legend()
    plt.savefig(align_path, dpi=300)
    plt.close()

    return shift, template_align

# Coincidence runs

runs = {
    0:     39510,
    10:    39511,
    20:    39512,
    30:    39514,
    40:    39515,
    50:    39516,
    60:    39517,
    70:    39518,
    80:    39519,
    90:    39521,
    100:   39522,
    110:   39523,
    120:   39525,
    130:   39526,
    140:   39527,
    150:   39528,
    155.7: 39529
}

runs = {f: r for f, r in runs.items() if f not in CFG['skip_fields']}

# trial and sigma run
for field, run in runs.items():

    file = coincidence_file(run)

    coin = pd.read_csv(file)

    # only zero

    if field == 0:

        shift, template_m8_2_align = deconvolve_align(
            coin['mean'],
            template_m8_2,
            f'{output_folder}/templates/coin_template_noalign.png',
            f'{output_folder}/templates/coin_template_align.png'
        )

        shift_mine, template_2081_align = deconvolve_align(
            coin['mean'],
            template_2081,
            f'{output_folder}/templates/coin_template_noalign_mine.png',
            f'{output_folder}/templates/coin_template_align_mine.png'
        )

        kernel = np.roll(
            template_m8_2,
            -int(np.argmax(np.abs(np.diff(template_m8_2))))
        )

        # same kernel construction, but from my own template (ch 2081)
        kernel_mine = np.roll(
            template_2081,
            -int(np.argmax(np.abs(np.diff(template_2081))))
        )

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # FFT data

        fft_0 = np.fft.rfft(coin['mean'])

        ax[0].plot(frequency_array, np.abs(fft_0))
        ax[0].set_xlim(0, 31.25)
        ax[0].set_yscale('log')
        ax[0].set_xlabel('frequency [MHz]')
        ax[0].set_ylabel('amplitude')
        ax[0].set_title('Coincidence 0kV')

        # FFT template

        fft_template = np.fft.rfft(kernel)
        fft_template_mine = np.fft.rfft(kernel_mine)

        ax[1].plot(frequency_array, np.abs(fft_template))
        ax[1].set_xlim(0, 31.25)
        ax[1].set_yscale('log')
        ax[1].set_xlabel('frequency [MHz]')
        ax[1].set_ylabel('amplitude')
        ax[1].set_title('Template')

        plt.tight_layout()
        plt.savefig(f'{output_folder}/fft/ffts.png', dpi=300)
        plt.close()

        # FFT of the two templates alone, compared

        fig, ax = plt.subplots()

        ax.plot(frequency_array, np.abs(fft_template), label='Francesca')
        ax.plot(frequency_array, np.abs(fft_template_mine), label='My template')
        ax.legend()
        ax.set_xlim(0, 31.25)
        ax.set_yscale('log')
        ax.set_xlabel('frequency [MHz]')
        ax.set_ylabel('amplitude')
        ax.set_title('Template FFT')
        fig.savefig(f'{output_folder}/fft/templates_fft.png', dpi=300)
        plt.close(fig)

        # gaussian filter

        L = fft_0 / fft_template
        L_mine = fft_0 / fft_template_mine

        fig, ax = plt.subplots()

        ax.scatter(
            frequency_array,
            np.abs(L),
            s=3,
            label='Francesca'
        )
        ax.scatter(
            frequency_array,
            np.abs(L_mine),
            s=3,
            label='My template'
        )
        ax.legend()
        ax.set_xlim(0, 31.25)
        ax.set_yscale('log')
        ax.set_xlabel('frequency [MHz]')
        ax.set_ylabel('L')
        fig.savefig(f'{output_folder}/fft/L_pre_filter.png', dpi=300)
        plt.close(fig)

        sigma_1 = 1

        sigma_2 = 2  # MHz

        sigma_3 = 3

        G_1 = np.exp(-frequency_array**2 / (2 * sigma_1**2))

        G_2 = np.exp(-frequency_array**2 / (2 * sigma_2**2))

        G_3 = np.exp(-frequency_array**2 / (2 * sigma_3**2))

        fig, ax = plt.subplots()

        ax.scatter(
            frequency_array,
            np.abs(G_1 * L),
            s=3,
            color='C0',
            label=rf'$\sigma_f = {sigma_1}$ MHz'
        )
        ax.scatter(
            frequency_array,
            np.abs(G_2 * L),
            s=3,
            color='C1',
            label=rf'$\sigma_f = {sigma_2}$ MHz'
        )
        ax.scatter(
            frequency_array,
            np.abs(G_3 * L),
            s=3,
            color='C2',
            label=rf'$\sigma_f = {sigma_3}$ MHz'
        )
        ax.scatter(
            frequency_array,
            np.abs(G_1 * L_mine),
            s=3,
            marker='x',
            color='C0',
            label=rf'$\sigma_f = {sigma_1}$ MHz (mine)'
        )
        ax.scatter(
            frequency_array,
            np.abs(G_2 * L_mine),
            s=3,
            marker='x',
            color='C1',
            label=rf'$\sigma_f = {sigma_2}$ MHz (mine)'
        )
        ax.scatter(
            frequency_array,
            np.abs(G_3 * L_mine),
            s=3,
            marker='x',
            color='C2',
            label=rf'$\sigma_f = {sigma_3}$ MHz (mine)'
        )
        ax.legend(fontsize=7)
        ax.set_xlim(0, 31.25)
        ax.set_yscale('log')
        ax.set_xlabel('frequency [MHz]')
        ax.set_ylabel('G*L')
        fig.savefig(f'{output_folder}/fft/GL.png', dpi=300)
        plt.close(fig)

        l_1 = np.fft.irfft(G_1 * L, n=len(time_array))

        l_2 = np.fft.irfft(G_2 * L, n=len(time_array))

        l_3 = np.fft.irfft(G_3 * L, n=len(time_array))

        fig, ax = plt.subplots()

        ax.scatter(
            time_array,
            l_1,
            s=3,
            label=rf'$\sigma_f = {sigma_1}$ MHz'
        )
        ax.scatter(
            time_array,
            l_2,
            s=3,
            label=rf'$\sigma_f = {sigma_2}$ MHz'
        )
        ax.scatter(
            time_array,
            l_3,
            s=3,
            label=rf'$\sigma_f = {sigma_3}$ MHz'
        )
        ax.legend()
        ax.set_xlabel('time [us]')
        ax.set_ylabel('Amplitude [ADC]')
        ax.set_xlim(0.25, 5)
        fig.savefig(f'{output_folder}/deconvolve_0kv_sigmas.png', dpi=300)
        plt.close(fig)

        # G1 deconvolved - log scale

        fig, ax = plt.subplots()

        ax.scatter(
            time_array,
            np.abs(l_1),
            s=3,
            label=rf'$\sigma_f = {sigma_1}$ MHz'
        )

        ax.set_xlim(0.25, 5)
        ax.set_ylim(0.1, 20)
        ax.set_yscale('log')
        ax.set_xlabel('time [us]')
        ax.set_ylabel('Amplitude [ADC]')
        ax.legend()

        fig.savefig(f'{output_folder}/deconvolve_0kv_G1_log.png', dpi=300)
        plt.close(fig)

# G1 deconvolved - all fields

fit_results = {}

initial_guesses = {
    't0': 1.0, 'Aslow': 0.2, 'Afast': 30.0, 
    'tau_slow': 1.4, 'tau_fast': 0.006, 
    'sigma': 0.06
}

l2_by_field = {}

for field, run in runs.items():

    file = coincidence_file(run)

    coin = pd.read_csv(file)

    fft_data = np.fft.rfft(coin['mean'])

    L = fft_data / fft_template

    l_2 = np.fft.irfft(
        G_2 * L,
        n=len(time_array)
    )

    # normalize by E_0 peak
    if field == 0:
        norm_factor = l_2.max()

    l_2 = l_2 / norm_factor

    fig, ax = plt.subplots()

    ax.scatter(
        time_array,
        np.abs(l_2),
        s=3
    )

    ax.set_xlim(0.25, 4)
    ax.set_ylim(0.01, 1)
    ax.set_yscale('log')
    ax.set_xlabel('time [us]')
    ax.set_ylabel('Amplitude [ADC]')
    ax.set_title(f'{field} kV')

    fig.savefig(
        f'{output_folder}/deconv/deconvolve_G1_{field}kv_log.png',
        dpi=300
    )

    plt.close(fig)

    # keep for the combined all-fields plots, built after the loop
    l2_by_field[field] = l_2.copy()

    FIT_MIN = 0.25    # us
    FIT_MAX = 2  # us

    mask = (time_array > FIT_MIN) & (time_array < FIT_MAX)

    x = time_array[mask]
    y = l_2[mask]  # ja normalizado uma unica vez pelo pico de E=0
    noise_region = time_array > 8
    yerr = np.std(l_2[noise_region] - np.mean(l_2[noise_region])) * np.ones_like(y)
    #yerr = 0.01

    cost = LeastSquares(x, y, yerr, model)

    m = Minuit(
        cost,
        **initial_guesses
    )


    m.limits['t0']       = (0.5, 1.5)
    m.limits['Aslow']    = (0, 0.5)
    m.limits['Afast']    = (0, 100)
    m.limits['tau_slow'] = (0.5, 2.5)
    m.limits['tau_fast'] = (0.001, 0.05)
    m.limits['sigma']    = (0.01, 0.1)

    m.fixed['tau_fast'] = True
    # if field == 0:
    #     m.interactive()
    #m.simplex()
    m.migrad()
    m.hesse()

    parameters = m.values
    errors = m.errors
    chisquared = m.fval
    ndof = len(x) - m.nfit
    chi2_ndof = chisquared / ndof

    fit_results[field] = m

    print(f'--- {field} kV ---')
    print(m)

    fig, ax = plt.subplots()

    ax.scatter(
        time_array,
        np.abs(l_2),
        s=3
    )

    legend = (
        rf"$\tau_s$ = {parameters['tau_slow']:.3f} ± {errors['tau_slow']:.3f}" "\n"
        rf"$\tau_f$ = {parameters['tau_fast']:.3f} ± {errors['tau_fast']:.3f}" "\n"
        rf"$A_s$ = {parameters['Aslow']:.3f} ± {errors['Aslow']:.3f}" "\n"
        rf"$A_f$ = {parameters['Afast']:.3f} ± {errors['Afast']:.3f}" "\n"
        rf"$\chi^2$/ndof = {chi2_ndof:.2f}"
    )


    ax.plot(
        x,
        model(x, *m.values),
        color='red',
        lw=1,
        label=legend
    )

    ax.set_xlim(0.25, 4)
    ax.set_ylim(0.01, 1)
    ax.set_yscale('log')
    ax.set_xlabel('time [us]')
    # add a shadow on fit region
    ax.axvspan(FIT_MIN, FIT_MAX, color='gray', alpha=0.5)
    ax.legend()
    ax.set_ylabel('Amplitude [ADC]')
    ax.set_title(f'{field} kV')

    fig.savefig(f'{output_folder}/fit/fit_{field}kv_log.png',
        dpi=300
    )

    plt.close(fig)

def make_all_fields_plots(l2_by_field, runs):
    """All deconvolved waveforms, same plot -- log and linear, with a
    zoomed inset on the peak region (0.7 - 1.3 us). Also saves a reduced
    version with fewer fields, easier to read."""

    INSET_XLIM = (0.7, 1.3)

    def plot_all_fields(fields_to_plot, yscale, out_path):
        fig, ax = plt.subplots()

        for field in fields_to_plot:
            l_2 = l2_by_field[field]
            y = np.abs(l_2) if yscale == 'log' else l_2
            ax.scatter(time_array, y, s=3, label=f'{field} kV')

        ax.set_xlim(0.25, 4)
        ax.set_xlabel('time [us]')
        ax.set_ylabel('Amplitude [ADC]')
        ax.legend(fontsize=6, ncol=2)

        axins = ax.inset_axes([0.55, 0.08, 0.4, 0.35])
        for field in fields_to_plot:
            l_2 = l2_by_field[field]
            y = np.abs(l_2) if yscale == 'log' else l_2
            axins.scatter(time_array, y, s=3)
        axins.set_xlim(*INSET_XLIM)

        if yscale == 'log':
            ax.set_ylim(0.01, 1)
            ax.set_yscale('log')
            axins.set_ylim(0.05, 1.1)
            axins.set_yscale('log')
        else:
            axins.set_ylim(-0.05, 1.05)

        ax.indicate_inset_zoom(axins, edgecolor='black')

        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    all_fields = list(runs.keys())
    subset_fields = [0, 10, 30, 60, 90, 120, 155.7]

    plot_all_fields(all_fields, 'log', f'{output_folder}/deconv/deconvolve_all_fields_log.png')
    plot_all_fields(all_fields, 'linear', f'{output_folder}/deconv/deconvolve_all_fields_linear.png')

    plot_all_fields(subset_fields, 'log', f'{output_folder}/deconv/deconvolve_fields_subset_log.png')
    plot_all_fields(subset_fields, 'linear', f'{output_folder}/deconv/deconvolve_fields_subset_linear.png')

# all deconvolved waveforms, same plot -- log and linear

make_all_fields_plots(l2_by_field, runs)

# parameters vs electric field

fields = np.array(list(fit_results.keys()))

DRIFT_LENGTH = 3.5  # m

efield = fields / DRIFT_LENGTH  # kV -> kV/m

params = ['t0', 'Aslow', 'Afast', 'tau_slow', 'tau_fast', 'sigma']

labels = {
    't0':       r'$t_0$ [us]',
    'Aslow':    r'$A_s$ [a.u.]',
    'Afast':    r'$A_f$ [a.u.]',
    'tau_slow': r'$\tau_s$ [us]',
    'tau_fast': r'$\tau_f$ [us]',
    'sigma':    r'$\sigma$ [us]',
}

for p in params:

    v = np.array([fit_results[f].values[p] for f in fields])

    e = np.array([fit_results[f].errors[p] for f in fields])

    fig, ax = plt.subplots()

    ax.errorbar(efield, v, yerr=e, fmt='o', ms=4)

    ax.set_xlabel('Electric Field [kV/m]')
    ax.set_ylabel(labels[p])

    fig.savefig(f'{output_folder}/parameters/{p}.png', dpi=300)

    plt.close(fig)

# ----------------------------------------------------------------------
# tau_slow vs campo eletrico -- modelo de Segreto, PRD 103, 043001 (2021)
# ----------------------------------------------------------------------
# Secao IV, Eq. 14: o triplete Ar2* e quenchado por reacao com ions Ar2+,
# cuja densidade depende do campo atraves do fator de recombinacao R(E).
# Dai a dependencia do tempo de decaimento lento com o campo aplicado:
#
#     tau_eff(E) = 1 / ( alpha + beta / (1 + k_E / E) )
#
# alpha = 1/tau_eff(0);  beta = k0+ A / B;  k_E = constante de recombinacao
# de Birks (Segreto obtem k_E = 0.075 +- 0.015 kV/cm ajustando os pontos de
# Lastoria et al.).  A forma abaixo, beta*E/(E + k_E), e algebricamente
# identica e nao diverge em E = 0.

SEGRETO_KE = 0.075   # kV/cm, valor de referencia do paper


def tau_eff_segreto(E_kVcm, alpha, beta, k_E):
    """Eq. 14 de Segreto 2021. E em kV/cm, alpha e beta em 1/us."""
    return 1.0 / (alpha + beta * E_kVcm / (E_kVcm + k_E))


# Segreto trabalha em kV/cm; os eixos daqui estao em kV/m
efield_kVcm = efield / 100.0

tau_s = np.array([fit_results[f].values['tau_slow'] for f in fields])
tau_e = np.array([fit_results[f].errors['tau_slow'] for f in fields])
tau_e = np.where(tau_e > 0, tau_e, np.nanmedian(tau_e[tau_e > 0]))

cost_tau = LeastSquares(efield_kVcm, tau_s, tau_e, tau_eff_segreto)

m_tau = Minuit(cost_tau, alpha=1.0 / tau_s[0], beta=0.1, k_E=SEGRETO_KE)
m_tau.limits['alpha'] = (0.1, 5.0)
m_tau.limits['beta'] = (0, 5.0)
m_tau.limits['k_E'] = (0.001, 2.0)
m_tau.migrad()
m_tau.hesse()

ndof_tau = len(fields) - m_tau.nfit
chi2_tau_raw = m_tau.fval / ndof_tau

# mesma inflacao de erro usada nos ajustes de waveform
scale_tau = np.sqrt(chi2_tau_raw) if chi2_tau_raw > 1 else 1.0
if scale_tau > 1:
    m_tau = Minuit(LeastSquares(efield_kVcm, tau_s, tau_e * scale_tau,
                                tau_eff_segreto), **m_tau.values.to_dict())
    m_tau.limits['alpha'] = (0.1, 5.0)
    m_tau.limits['beta'] = (0, 5.0)
    m_tau.limits['k_E'] = (0.001, 2.0)
    m_tau.migrad()
    m_tau.hesse()

a_, b_, kE_ = m_tau.values['alpha'], m_tau.values['beta'], m_tau.values['k_E']
tau_0 = 1.0 / a_
tau_inf = 1.0 / (a_ + b_)

print('\n=== tau_slow(E): modelo de Segreto 2021, Eq. 14 ===')
print(m_tau)
print(f'chi2/ndof bruto = {chi2_tau_raw:.3g}  (escala de erro {scale_tau:.3g})')
print(f'tau_eff(0)    = {tau_0 * 1000:.1f} ns        [Segreto/Lastoria: ~1420 ns]')
print(f'tau_eff(inf)  = {tau_inf * 1000:.1f} ns      [Segreto/Lastoria: ~1240 ns]')
print(f'queda relativa= {(tau_inf / tau_0 - 1) * 100:+.1f}%   '
      f'[Segreto/Lastoria: ~-13%]')
print(f'k_E           = {kE_:.4f} +- {m_tau.errors["k_E"]:.4f} kV/cm   '
      f'[Segreto: 0.075 +- 0.015]')

# --- figura
E_smooth = np.linspace(0, efield.max() * 1.05, 400)

fig, ax = plt.subplots()

ax.errorbar(efield, tau_s * 1000, yerr=tau_e * 1000, fmt='o', ms=4,
            label='este trabalho')
ax.plot(E_smooth, tau_eff_segreto(E_smooth / 100.0, a_, b_, kE_) * 1000,
        'r-', lw=1.2,
        label=(rf'Segreto Eq. 14: $k_\mathcal{{E}}$ = {kE_:.3f} $\pm$ '
               rf'{m_tau.errors["k_E"]:.3f} kV/cm'))
ax.axhline(tau_0 * 1000, color='gray', ls=':', lw=0.8)
ax.axhline(tau_inf * 1000, color='gray', ls=':', lw=0.8)

ax.text(0.98, 0.95,
        rf'$\tau_{{eff}}(0)$ = {tau_0*1000:.0f} ns' '\n'
        rf'$\tau_{{eff}}(\infty)$ = {tau_inf*1000:.0f} ns' '\n'
        rf'queda = {(tau_inf/tau_0 - 1)*100:+.1f}%' '\n'
        rf'$\chi^2$/ndof = {m_tau.fval / ndof_tau:.2f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=8)

ax.set_xlabel('Electric Field [kV/m]')
ax.set_ylabel(r'$\tau_{slow}$ [ns]')
ax.legend(fontsize=8, loc='lower left')

fig.savefig(f'{output_folder}/parameters/tau_slow_vs_field_segreto.png', dpi=300)
plt.close(fig)

# --- versao com k_E fixo no valor do paper, para comparar
m_fix = Minuit(LeastSquares(efield_kVcm, tau_s, tau_e * scale_tau,
                            tau_eff_segreto),
               alpha=a_, beta=b_, k_E=SEGRETO_KE)
m_fix.limits['alpha'] = (0.1, 5.0)
m_fix.limits['beta'] = (0, 5.0)
m_fix.fixed['k_E'] = True
m_fix.migrad()
m_fix.hesse()
print(f'\ncom k_E fixo em {SEGRETO_KE} kV/cm: '
      f'tau_eff(0) = {1000 / m_fix.values["alpha"]:.1f} ns, '
      f'chi2/ndof = {m_fix.fval / (len(fields) - m_fix.nfit):.2f} '
      f'(livre: {m_tau.fval / ndof_tau:.2f})')


# likelihood profiles for tau_slow

PROFILE_FIELDS = [0, 50, 100]

# grade grossa: forma global do perfil, 0.5 a 2.0 us em passos de 0.1
TAU_GRID_COARSE = np.linspace(0.5, 2.0, 16)

# grade fina em torno do minimo: e ela que resolve dchi2 = 1.
# O passo se adapta ao erro do Hesse (~sigma/5); um passo fixo de 5 ns nao
# resolve nada quando o ajuste e muito preciso.
TAU_FINE_NSIGMA = 5       # meia-largura da janela fina, em sigmas do Hesse
TAU_FINE_PER_SIGMA = 5    # pontos por sigma dentro da janela
TAU_FINE_STEP_MIN = 0.0002   # us, piso do passo
TAU_FINE_STEP_MAX = 0.010    # us, teto do passo

os.makedirs(f'{output_folder}/profiles', exist_ok=True)


def build_profile_grid(best, err):
    """Grade grossa + grade fina ao redor do minimo, unidas e ordenadas.

    Com passo de 0.1 us nenhum ponto cai dentro de dchi2 <= 1 (o erro do Hesse
    e ~0.02 us), entao a grade grossa serve so para a forma global; a janela
    fina de +-5 sigma com passo de 5 ns e o que permite ler o intervalo.
    """
    sigma = err if (np.isfinite(err) and err > 0) else 0.02

    half = min(max(TAU_FINE_NSIGMA * sigma, 0.002), 0.50)
    step = min(max(sigma / TAU_FINE_PER_SIGMA, TAU_FINE_STEP_MIN),
               TAU_FINE_STEP_MAX)

    lo = max(TAU_GRID_COARSE[0], best - half)
    hi = min(TAU_GRID_COARSE[-1], best + half)

    fine = np.arange(lo, hi + 0.5 * step, step)
    grid = np.union1d(TAU_GRID_COARSE, fine)

    return grid, (lo, hi)


def profile_1sigma(tau_scan, dchi2, ok, best):
    """Cruzamentos de dchi2 = 1 por interpolacao linear entre pontos vizinhos."""
    t = np.asarray(tau_scan)[ok]
    d = np.asarray(dchi2)[ok]
    lo = hi = np.nan
    for i in range(len(t) - 1):
        d0, d1 = d[i], d[i + 1]
        if (d0 - 1) * (d1 - 1) <= 0 and d0 != d1:
            xc = t[i] + (1 - d0) * (t[i + 1] - t[i]) / (d1 - d0)
            if xc <= best:
                lo = xc
            elif np.isnan(hi):
                hi = xc
    return lo, hi


fig_all, ax_all = plt.subplots()

for pfield in PROFILE_FIELDS:

    if pfield not in fit_results:
        print(f'[profile] {pfield} kV nao esta em fit_results, pulando')
        continue

    m_p = fit_results[pfield]
    best = m_p.values['tau_slow']
    err = m_p.errors['tau_slow']

    grid, fine_window = build_profile_grid(best, err)

    # para cada tau_slow fixo, reminimiza todos os outros parametros
    tau_scan, dchi2, ok = m_p.mnprofile(
        'tau_slow',
        grid=grid,
        subtract_min=True
    )

    lo, hi = profile_1sigma(tau_scan, dchi2, ok, best)

    # --- figura: faixa completa (symlog) + zoom na janela fina
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, zoom in zip(axes, (False, True)):

        ax.plot(tau_scan, dchi2, 'o-', ms=3, lw=1.0, color='C0')

        if (~ok).any():
            ax.scatter(tau_scan[~ok], dchi2[~ok], s=25, color='red',
                       label='minimizacao falhou', zorder=3)

        for lvl in (1, 4, 9):
            ax.axhline(lvl, color='gray', ls='--', lw=0.8)

        ax.axvline(best, color='red', lw=1,
                   label=rf'$\hat{{\tau}}_s$ = {best:.4f} $\pm$ {err:.4f} us (Hesse)')

        if np.isfinite(lo) and np.isfinite(hi):
            ax.axvspan(lo, hi, color='red', alpha=0.12,
                       label=rf'perfil 1$\sigma$: $-${best - lo:.4f} / $+${hi - best:.4f} us')

        ax.set_xlabel(r'$\tau_{slow}$ [us]')
        ax.set_ylabel(r'$\Delta \chi^2$')

        if zoom:
            ax.set_xlim(*fine_window)
            ax.set_ylim(0, 12)
            ax.set_title('zoom', fontsize=10)
            ax.legend(fontsize=7)
        else:
            ax.set_xlim(TAU_GRID_COARSE[0], TAU_GRID_COARSE[-1])
            ax.set_yscale('symlog', linthresh=1)
            ax.set_title(f'{pfield} kV  ({pfield / DRIFT_LENGTH:.1f} kV/m)')

    fig.tight_layout()
    fig.savefig(f'{output_folder}/profiles/profile_tau_slow_{pfield}kv.png', dpi=300)
    plt.close(fig)

    # --- figura combinada (so o zoom, centrada em cada minimo)
    ax_all.plot(tau_scan - best, dchi2, 'o-', ms=3, lw=1.0,
                label=f'{pfield / DRIFT_LENGTH:.1f} kV/m')

    if np.isfinite(lo) and np.isfinite(hi):
        asym = ((hi - best) - (best - lo)) / (0.5 * (hi - lo)) * 100
        print(f'[profile] {pfield} kV: tau_slow = {best:.4f} '
              f'(-{best - lo:.4f} / +{hi - best:.4f}) us [perfil], '
              f'Hesse +-{err:.4f}, assimetria {asym:+.0f}%, '
              f'pontos na grade {len(grid)}, invalidos {(~ok).sum()}')
    else:
        print(f'[profile] {pfield} kV: perfil nao cruza dchi2=1 '
              f'(lo={lo}, hi={hi}); best={best:.4f} err={err:.4f} '
              f'-- provavelmente no limite')

for lvl in (1, 4, 9):
    ax_all.axhline(lvl, color='gray', ls='--', lw=0.8)

ax_all.set_xlabel(r'$\tau_{slow} - \hat{\tau}_{slow}$ [us]')
ax_all.set_ylabel(r'$\Delta \chi^2$')
ax_all.set_xlim(-0.15, 0.15)
ax_all.set_ylim(0, 12)
ax_all.legend(fontsize=8)
fig_all.savefig(f'{output_folder}/profiles/profile_tau_slow_all.png', dpi=300)
plt.close(fig_all)


# integrals

I_slow = np.array([fit_results[f].values['Aslow'] * fit_results[f].values['tau_slow'] for f in fields])

I_fast = np.array([fit_results[f].values['Afast'] * fit_results[f].values['tau_fast'] for f in fields])

I_total = I_slow + I_fast

for name, v in [('integral_slow', I_slow / I_slow[0]),
                ('integral_fast', I_fast / I_fast[0]),
                ('integral_total', I_total / I_total[0]),
                ('fast_over_total', I_fast / I_total)]:

    fig, ax = plt.subplots()

    ax.scatter(efield, v, s=15)

    ax.set_xlabel('Electric Field [kV/m]')
    ax.set_ylabel(name.replace('_', ' '))

    fig.savefig(f'{output_folder}/parameters/{name}.png', dpi=300)

    plt.close(fig)