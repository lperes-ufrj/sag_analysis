"""Deconvolucao preliminar da coincidencia em campo nulo (E = 0).

Bloco que vivia no inicio de todos os fitting-bertolini*.py e nao entrava no
loop de fits: alinhamento template x coincidencia (correlacao), FFT dos dados
e dos templates (Francesca e proprio), L = V/H, filtro gaussiano G com
sigma_f = 1, 2 e 2.62 MHz, e a irfft resultante.

Usa a mesma configuracao de modulos/arquivos do fitting-bertolini-avg.py
(rodada de 09/set/2026). Como antes, a coincidencia usada e a do segundo
canal do modulo (CHANNELS[1]) e o template da Francesca correspondente.

    python deconvolution-preliminary.py [M8|M6|M5]

Saida em plots_deconvolution_preliminary/<modulo>/.
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dunestyle.matplotlib as dunestyle  # noqa: F401  (estilo DUNE no import)

# caminhos ancorados no proprio script: funciona de qualquer diretorio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # sag_analysis/fitting
REPO_DIR = os.path.dirname(SCRIPT_DIR)                    # sag_analysis

SAMPLE = 16  # ns

TIME_WINDOW = 1024 * SAMPLE / 1000  # us

NYQUIST_FREQUENCY = 1 / (2 * SAMPLE / 1000)   # = 31.25 → MHz

FREQUENCY_WINDOW = 1 / TIME_WINDOW

# modulo (arapuca) analisado, com os dois canais combinados:
#   M8 (2080/2081), M6 (2060/2061), M5 (2050/2051).
#
# Nao se escolhe mais editando aqui: o script pergunta ao iniciar (ver
# escolhe_modulo(), logo abaixo do MODULE_CONFIG). Para rodar sem o menu,
# passe o modulo na linha de comando:
#     python fitting-bertolini-avg.py M6
MENU_MODULES = ('M8', 'M5', 'M6')
MODULE_DEFAULT = 'M8'

# por modulo: canais, templates da Francesca (1 -> canal par, 2 -> canal
# impar), rodada de coincidencia (subpasta de coincidence/selected_waveforms/,
# a mesma publicada no GitHub lperes-ufrj/sag_analysis) e campos a pular.

# --- rodada de 09/set/2026 (Leo, repo lperes-ufrj/sag_analysis) ------------
# Cada reprocessamento vive numa subpasta com o timestamp e tem nome de
# arquivo proprio. A definicao da coincidencia saiu do NOME do arquivo e foi
# para saved_coincidences/<timestamp>/coincidence_scan_<timestamp>.txt.
#
# M7 + M8 estao em 20260909_125014. O gatilho e o mesmo de antes
# (2030-2031-2040-2041 vs 2050-2051-2060-2061, salvando 2070/2071/2080/2081),
# MAS entrou um corte novo ANTES da coincidencia: limiar de amplitude por
# canal e por run, calibrado para igualar a taxa entre canais
# (analysis/RateAnalysis_data_292kV/equalized_run_*_thresholds.txt).
#
# Isso muda o dado, nao so o nome: o mesmo run 43386 / canal 2080 vai de
# 2162 para 1747 ADC no pico. NAO misturar rodadas -- ou tudo novo, ou tudo
# velho, dentro de um mesmo conjunto de campos.
# M7 + M8: gatilho 2030-2031-2040-2041 vs 2050-2051-2060-2061
SCAN_M8 = '20260909_125014'

# M5 + M6: gatilho 2070-2071-2080-2081 vs 2010-2011-2020-2021. Note que os
# dois conjuntos se disparam mutuamente -- o M8 e gatilhado por M5/M6 e vice
# versa -- entao cada canal analisado segue fora do proprio gatilho, mas as
# amostras de eventos das duas rodadas NAO sao as mesmas.
SCAN_LOW = '20260909_213507'

# nome do arquivo na rodada; {ch}, {run} e {scan} sao preenchidos em
# coincidence_file() ({scan} = o proprio subdir).
PATTERN_NEW = 'channel_{ch}_coincidence_scan_run_0{run}_{scan}.csv'

MODULE_CONFIG = {
    # fit_max: teto da janela de ajuste (FIT_MIN continua fixo em 0.25 us
    # para todos). Valor por arapuca, do scan de debug em FIT_MAX
    # (debug_scan_fitwindow/tau_slow_vs_fitmax_*.png): onde tau_slow(E=0)
    # comeca a estabilizar em vez de continuar caindo/subindo com o teto.
    #   M8 ~ 2.95 us (entre 2.9 e 3.0)
    #   M5 ~ 2.56 us (entre 2.5 e 2.6)
    #   M6 : usa o mesmo valor do M5 por enquanto -- scan de M6 proprio
    #        ainda nao feito
    'M8': dict(channels=(2080, 2081),
               templates=('template_42156_M8_1.txt', 'template_42156_M8_2.txt'),
               subdir=SCAN_M8, pattern=PATTERN_NEW, skip_fields=(),
               fit_max=2.95),
    # t0 com janela larga: ver a nota em T0_LIMITS, mais abaixo
    'M6': dict(channels=(2060, 2061),
               templates=('template_40808_M6_1.txt', 'template_40808_M6_2.txt'),
               subdir=SCAN_LOW, pattern=PATTERN_NEW, skip_fields=(),
               t0_limits=(0.9, 1.4), fit_max=2.56),
    'M5': dict(channels=(2050, 2051),
               templates=('template_41522_M5_1.txt', 'template_41522_M5_2.txt'),
               subdir=SCAN_LOW, pattern=PATTERN_NEW, skip_fields=(),
               fit_max=2.56),
}

def escolhe_modulo():
    """Modulo a analisar: argumento de linha de comando, senao menu.

    Sem terminal interativo (pipe, nohup, batch) cai no MODULE_DEFAULT em
    vez de travar num input() que ninguem vai responder."""

    if len(sys.argv) > 1:
        arg = sys.argv[1].strip().upper()

        if arg not in MODULE_CONFIG:
            raise SystemExit(
                f'modulo invalido: {sys.argv[1]!r}. '
                f'Validos: {", ".join(MODULE_CONFIG)}')

        return arg

    if not sys.stdin.isatty():
        print(f'[aviso] sem terminal interativo; usando {MODULE_DEFAULT}')
        return MODULE_DEFAULT

    print('\nModulo a analisar (media dos dois canais do modulo):\n')

    for i, mod in enumerate(MENU_MODULES, 1):
        a, b = MODULE_CONFIG[mod]['channels']
        print(f'  {i}) {mod}   canais {a} + {b}')

    print()

    while True:
        resp = input(f'Escolha 1-{len(MENU_MODULES)}, ou o nome do modulo '
                     f'[enter = {MODULE_DEFAULT}]: ').strip().upper()

        if not resp:
            return MODULE_DEFAULT

        if resp.isdigit() and 1 <= int(resp) <= len(MENU_MODULES):
            return MENU_MODULES[int(resp) - 1]

        # aceita qualquer modulo do config pelo nome
        if resp in MODULE_CONFIG:
            return resp

        print('  opcao invalida')


MODULE = escolhe_modulo()

print(f'\n>>> modulo: {MODULE}  '
      f'(canais {" + ".join(str(c) for c in MODULE_CONFIG[MODULE]["channels"])})\n')

CFG = MODULE_CONFIG[MODULE]


def coincidence_file(run, ch):
    sub = f'{CFG["subdir"]}/' if CFG['subdir'] else ''

    return (REPO_DIR + '/coincidence/selected_waveforms/'
            + sub + CFG['pattern'].format(ch=ch, run=run,
                                          scan=CFG['subdir']))


output_folder = os.path.join(SCRIPT_DIR, f'plots_deconvolution_preliminary/{MODULE}')

for folder in ['templates', 'fft']:
    os.makedirs(f'{output_folder}/{folder}', exist_ok=True)

# templates francesca

time_array = np.arange(0, 1024 * SAMPLE, SAMPLE) / 1000  # us

frequency_array = np.fft.rfftfreq(len(time_array), SAMPLE / 1000)  # MHz

template_m8_1 = np.loadtxt(
    REPO_DIR + f'/calib/output/data/templates/francesca_template/{CFG["templates"][0]}'
)

template_m8_2 = np.loadtxt(
    REPO_DIR + f'/calib/output/data/templates/francesca_template/{CFG["templates"][1]}'
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

# templates proprios (.npy) so existem para o M8; nos outros modulos os
# plots de diagnostico *_mine repetem o template da Francesca do modulo
if MODULE != 'M8':
    template_2080 = template_m8_1
    template_2081 = template_m8_2

CHANNELS = CFG['channels']

RUN_0 = 39510   # campo nulo (mesmo run para todos os modulos)


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


# --- deconvolucao em E = 0 ---------------------------------------------

file = coincidence_file(RUN_0, CHANNELS[1])

coin = pd.read_csv(file)


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

sigma_3 = 2.62

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
