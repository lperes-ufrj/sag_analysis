"""fitting-bertolini.py com os dois canais do M8 combinados.

Unica diferenca: o espectro ajustado e a media de channel_2080 e
channel_2081, e o kernel e a media dos kernels dos templates
correspondentes (M8_1 <-> 2080, M8_2 <-> 2081). O resto e identico.
"""
import os
import sys

import matplotlib.pyplot as plt
import scipy as sci
import pandas as pd
import numpy as np
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


# --- incerteza usada no fit ------------------------------------------------
# 'csv'   : coluna 'mean_uncertainty' do arquivo de coincidencia, combinada
#           entre os dois canais do modulo. E o erro da MEDIA por amostra
#           (sigma_evento / sqrt(N_eventos)) e vale ~0.65% do sinal em toda a
#           regiao do pulso, ou seja peso relativo constante -- equivalente a
#           ajustar em escala log, que e como os dados sao olhados. So existe
#           nos arquivos reprocessados; nos antigos cai sozinho em 'noise'.
#
# 'noise' : comportamento antigo -- std do 'mean' em t > 8 us, constante.
#           CUIDADO: essa regiao NAO e ruido. O 'mean' la esta em ~-150 ADC
#           (-4% do pico) e ainda relaxando, entao esse numero mede a
#           inclinacao do undershoot, nao a flutuacao estatistica.
YERR_MODE = 'noise'

# Piso somado em quadratura a mean_uncertainty, medido no proprio dado na
# regiao de pre-pulso. Antes de t0 o pulso nao comecou, a dispersao entre
# eventos e ~zero e mean_uncertainty cai para ~0.04 ADC -- 500x menor que no
# pico. Sem o piso essas amostras dominam o chi2 e travam o t0 no pedestal.
PRE_PULSE_MAX = 0.9   # us


def fit_uncertainty(waveform, waveform_err, mask):
    """yerr por amostra dentro da janela de fit (waveform ja normalizada)."""

    if YERR_MODE == 'csv' and waveform_err is not None:
        sigma_pre = np.std(waveform[time_array < PRE_PULSE_MAX])
        return np.sqrt(waveform_err[mask]**2 + sigma_pre**2)

    noise_region = time_array > 8

    return np.std(
        waveform[noise_region] - np.mean(waveform[noise_region])
    ) * np.ones(int(mask.sum()))


output_folder = os.path.join(SCRIPT_DIR, f'plots_bertolini_avg/{MODULE}')

for folder in ['templates', 'deconv', 'fits', 'fit', 'parameters']:
    os.makedirs(f'{output_folder}/{folder}', exist_ok=True)

from scipy.signal import fftconvolve


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

# templates francesca

time_array = np.arange(0, 1024 * SAMPLE, SAMPLE) / 1000  # us

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

# kernel do fit: template da Francesca (canal 2081 -> M8_2), causal e de
# area unitaria. E este objeto, e nao o template alinhado por correlacao,
# que entra na convolucao do model().

# canal 2080 <-> template M8_1, canal 2081 <-> template M8_2.
# O fit usa a media dos dois espectros do M8; como cada kernel ja tem area
# unitaria, a media dos dois tambem tem.

CHANNELS = CFG['channels']


def coincidence_mean(run):
    """Media dos espectros de coincidencia dos dois canais do modulo, e o
    erro dessa media.

    Os dois canais sao independentes, entao para a media aritmetica
    sigma = sqrt(sum sigma_i^2) / N. Devolve sigma = None se algum dos
    arquivos nao trouxer a coluna 'mean_uncertainty' (arquivos antigos) --
    nesse caso o fit cai no modo 'noise'."""

    dfs = [pd.read_csv(coincidence_file(run, ch)) for ch in CHANNELS]

    mean = np.mean([df['mean'].to_numpy() for df in dfs], axis=0)

    if all('mean_uncertainty' in df.columns for df in dfs):
        sigma = np.sqrt(np.sum(
            [df['mean_uncertainty'].to_numpy()**2 for df in dfs], axis=0
        )) / len(dfs)
    else:
        sigma = None

    return mean, sigma


FIT_KERNEL = 0.5 * (make_causal_kernel(template_m8_1)
                    + make_causal_kernel(template_m8_2))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

ax[0].scatter(time_array, template_m8_2, s=3, label='template bruto')
ax[0].scatter(time_array, FIT_KERNEL, s=3, label='kernel causal (area 1)')
ax[0].set_xlim(0, 6)
ax[0].set_xlabel('time [us]')
ax[0].set_ylabel('amplitude')
ax[0].legend(fontsize=8)

ax[1].scatter(time_array, np.abs(FIT_KERNEL), s=3)
ax[1].set_xlim(0, 6)
ax[1].set_yscale('log')
ax[1].set_xlabel('time [us]')
ax[1].set_ylabel('|kernel|')
ax[1].set_title(f'area = {FIT_KERNEL.sum() * SAMPLE / 1000:.3f}', fontsize=9)

fig.tight_layout()
fig.savefig(f'{output_folder}/templates/fit_kernel.png', dpi=300)
plt.close(fig)

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
    # Scan de HV alto (set/2026). Cada HV tem dois runs; usamos o primeiro
    # de cada par -- os segundos sao 43381 (240), 43384 (205), 43387 (270),
    # 43390 (154). Ordem crescente em campo para o seeding sequencial do fit.
    #
    # 41523 = 292 kV e o campo mais alto do scan (0.834 kV/cm). O HV de cada
    # run vem do nome dos ROOT de entrada, listados no
    # coincidence_scan_<timestamp>.txt da rodada.
    #
    # Os runs 39500-39508 (scan DESCENDO, 155.7 -> 20 kV) ficam de fora de
    # proposito: todo HV deles ja aparece no scan subindo, entao nao trazem
    # campo novo -- so serviriam como teste de histerese, que nao e o objetivo
    # aqui. Se um dia entrarem, a chave do dict precisa deixar de ser o campo.
    154:   43389,
    155.7: 39529,
    205:   43383,
    240:   43380,
    270:   43386,
    292:   41523
}

runs = {f: r for f, r in runs.items() if f not in CFG['skip_fields']}

# Loop principal: fit forward-fold campo a campo
# (a deconvolucao preliminar em E = 0 foi para deconvolution-preliminary.py)

fit_results = {}

# chi2/ndof e flags do Minuit por campo, guardados para o dump em txt no
# fim do script (o loop ja calcula, so nao guardava)
fit_stats = {}

# NB: com FIT_KERNEL de area unitaria, Aslow/Afast estao na escala da luz
# emitida (dado normalizado pelo pico de E=0), nao na escala do dado.

# --- estrategia de ajuste --------------------------------------------------
# Deixar tudo livre nao converge de forma estavel. Tres amarras:
#
#  1. tau_fast FIXO. 6 ns << 16 ns de amostragem: a componente rapida ocupa
#     menos de uma amostra, so o produto A_f*tau_f e observavel. Deixando
#     livre, o fit empurrava tau_fast para o limite inferior (1 ns) em
#     metade dos campos, com Hesse degenerado e erro de I_fast explodindo.
#
#  2. t0 com limites estreitos. E o mesmo trigger em todos os runs; nos
#     testes variou 1.00800 -> 1.00727 entre 0 e 155.7 kV, ou seja 0.7 ns,
#     bem abaixo de uma amostra. A janela abaixo e ~+-0.6 amostra.
#
#  3. Chutes semeados sequencialmente: E=0 parte do ajuste interativo, cada
#     campo seguinte parte do resultado do campo anterior. tau_slow, A_s e
#     A_f variam suavemente com o campo, entao o campo anterior e sempre um
#     chute melhor que um valor fixo.

TAU_FAST_FIXED = 0.006   # us -- singlet do LAr, valor de literatura.
                         # Nao e mensuravel com 16 ns/amostra; ancorar aqui
                         # torna I_fast = A_f*tau_f comparavel a outros
                         # trabalhos, em vez de herdar o vies da grade.

T0_CENTER = 1.008        # us, valor do ajuste em E=0 no M8

T0_HALF = 0.010          # us, meia-janela padrao (~0.6 amostra)

# Janela permitida para t0, por modulo (chave 't0_limits' no MODULE_CONFIG;
# default = T0_CENTER +- T0_HALF).
#
# M8 e M5 sobem no mesmo tick -- em E = 0 os dois cruzam 10% do pico no tick
# 60-61 e picam no tick 91-92 -- entao a janela estreita original serve para
# os dois.
#
# O M6 nao: cruza 10% no tick 64 (~60 ns depois) e pica no tick 76-77, ou
# seja 14-15 ticks ANTES. Subida-ate-o-pico de ~210 ns contra ~480 ns do
# M8/M5: e resposta mais rapida, nao atraso de cabo. A janela dele vai de
# 0.9 a 1.4 us -- larga de proposito, para deixar o fit encontrar sozinho
# onde o t0 quer ficar em vez de encostar num limite apertado.
T0_LIMITS = CFG.get('t0_limits', (T0_CENTER - T0_HALF, T0_CENTER + T0_HALF))

# Janela do fit: FIT_MIN fixo, FIT_MAX por modulo (chave 'fit_max' no
# MODULE_CONFIG; default = 3 us para quem nao tem valor proprio). Ver a nota junto do MODULE_CONFIG, mais acima.
FIT_MIN = 0.25    # us
FIT_MAX = CFG.get('fit_max', 3)  # us

initial_guesses = {
    't0': T0_CENTER,
    'Aslow': 1.16,
    # 63.3 * 0.00162 / 0.006 = 17.1: preserva o produto A_f*tau_f do ajuste
    # interativo, que foi feito com tau_fast = 1.62 ns
    'Afast': 17.1,
    'tau_slow': 1.03,
    'tau_fast': TAU_FAST_FIXED,
}

LIMITS = {
    't0':       T0_LIMITS,
    'Aslow':    (0, 5),
    'Afast':    (0, 100),
    # tau_slow deve CAIR com o campo; o piso fica folgado de proposito para
    # nao truncar os campos altos (nos testes chegou a 0.98 us)
    'tau_slow': (0.7, 1.6),
    'tau_fast': (0.001, 0.05),
}

# semente do ajuste sequencial, atualizada a cada campo
seed = dict(initial_guesses)

l2_by_field = {}

for field, run in runs.items():

    waveform, waveform_err = coincidence_mean(run)

    # normalize by E_0 peak
    if field == 0:
        norm_factor = waveform.max()

    waveform = waveform / norm_factor

    if waveform_err is not None:
        waveform_err = waveform_err / norm_factor

    fig, ax = plt.subplots()

    ax.scatter(
        time_array,
        np.abs(waveform),
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
    l2_by_field[field] = waveform.copy()

    mask = (time_array > FIT_MIN) & (time_array < FIT_MAX)

    x = time_array[mask]
    y = waveform[mask]  
    yerr = fit_uncertainty(waveform, waveform_err, mask)

    cost = LeastSquares(x, y, yerr, model)

    m = Minuit(
        cost,
        **seed
    )

    for _name, _lim in LIMITS.items():
        m.limits[_name] = _lim

    m.fixed['tau_fast'] = True

    # descomente para reabrir o painel interativo num campo especifico
    if field == 0:
        m.interactive()

    m.migrad()
    m.hesse()

    # semeia o proximo campo com este resultado; se o ajuste falhou, mantem
    # a semente anterior em vez de propagar um minimo ruim
    if m.valid:
        seed = m.values.to_dict()
    else:
        print(f'[aviso] {field} kV: minimo invalido, semente nao atualizada')

    parameters = m.values
    errors = m.errors

    chisquared = m.fval
    ndof = len(x) - m.nfit
    chi2_ndof = chisquared / ndof

    fit_results[field] = m

    fit_stats[field] = {
        'chi2': chisquared,
        'ndof': ndof,
        'chi2_ndof': chi2_ndof,
        'valid': bool(m.valid),
        'accurate': bool(m.accurate),
    }

    print(f'--- {field} kV ---')
    print(m)

    fig, ax = plt.subplots()

    ax.scatter(
        time_array,
        np.abs(waveform),
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

    ax.set_xlim(0.25, 5)
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
    subset_fields = [f for f in [0, 10, 30, 60, 90, 120, 155.7]
                     if f in l2_by_field]

    plot_all_fields(all_fields, 'log', f'{output_folder}/deconv/deconvolve_all_fields_log.png')
    plot_all_fields(all_fields, 'linear', f'{output_folder}/deconv/deconvolve_all_fields_linear.png')

    plot_all_fields(subset_fields, 'log', f'{output_folder}/deconv/deconvolve_fields_subset_log.png')
    plot_all_fields(subset_fields, 'linear', f'{output_folder}/deconv/deconvolve_fields_subset_linear.png')

# all deconvolved waveforms, same plot -- log and linear

make_all_fields_plots(l2_by_field, runs)

# parameters vs electric field

fields = np.array(list(fit_results.keys()))

DRIFT_LENGTH = 3.5  # m

efield = fields / DRIFT_LENGTH / 100  # kV -> kV/cm

params = ['t0', 'Aslow', 'Afast', 'tau_slow', 'tau_fast']

labels = {
    't0':       r'$t_0$ [us]',
    'Aslow':    r'$A_s$ [a.u.]',
    'Afast':    r'$A_f$ [a.u.]',
    'tau_slow': r'$\tau_s$ [us]',
    'tau_fast': r'$\tau_f$ [us]',
}

for p in params:

    v = np.array([fit_results[f].values[p] for f in fields])

    e = np.array([fit_results[f].errors[p] for f in fields])

    fig, ax = plt.subplots()

    ax.errorbar(efield, v, yerr=e, fmt='o', ms=4)

    ax.set_xlabel('Electric Field [kV/cm]')
    ax.set_ylabel(labels[p])

    fig.savefig(f'{output_folder}/parameters/{p}.png', dpi=300)

    plt.close(fig)


# ----------------------------------------------------------------------
# integrais das componentes: I_i = A_i * tau_i
# ----------------------------------------------------------------------
# Para A*exp(-(t-t0)/tau)*theta(t-t0), a integral no tempo e exatamente
# A*tau. Como FIT_KERNEL tem area unitaria, a convolucao preserva area:
# I_i e a integral da componente i da luz emitida, nas mesmas unidades do
# dado (normalizado pelo pico de E=0).
#
# Erro com o termo de correlacao -- A e tau sao fortemente anticorrelacionados
# no fit, ignorar a covariancia superestima o erro por um fator grande:
#   var(A*tau) = (tau*sA)^2 + (A*stau)^2 + 2*A*tau*cov(A,tau)


def integral_and_error(m, name_A, name_tau):
    """I = A*tau e seu erro, com o termo cruzado da matriz de covariancia."""

    A = m.values[name_A]
    tau = m.values[name_tau]

    sA = m.errors[name_A]
    stau = m.errors[name_tau]

    try:
        cov = m.covariance[name_A, name_tau]
    except Exception:
        cov = 0.0

    var = (tau * sA)**2 + (A * stau)**2 + 2.0 * A * tau * cov

    return A * tau, np.sqrt(max(var, 0.0))


I_slow, I_slow_err = np.array([
    integral_and_error(fit_results[f], 'Aslow', 'tau_slow') for f in fields
]).T

I_fast, I_fast_err = np.array([
    integral_and_error(fit_results[f], 'Afast', 'tau_fast') for f in fields
]).T

I_total = I_slow + I_fast

# cov(I_slow, I_fast) ignorada aqui; as duas componentes sao pouco
# correlacionadas comparado a correlacao A-tau dentro de cada uma
I_total_err = np.sqrt(I_slow_err**2 + I_fast_err**2)

# fracao rapida: f = I_fast / I_total. Como I_total = I_fast + I_slow,
# df/dI_fast = I_slow/I_total^2  e  df/dI_slow = -I_fast/I_total^2
f_fast = I_fast / I_total
f_fast_err = np.sqrt(
    (I_slow * I_fast_err)**2 + (I_fast * I_slow_err)**2
) / I_total**2


def plot_vs_field(y, yerr, ylabel, out_name, hline=None):

    fig, ax = plt.subplots()

    ax.errorbar(efield, y, yerr=yerr, fmt='o', ms=4)

    if hline is not None:
        ax.axhline(hline, color='gray', ls=':', lw=0.8)

    ax.set_xlabel('Electric Field [kV/cm]')
    ax.set_ylabel(ylabel)

    fig.savefig(f'{output_folder}/parameters/{out_name}.png', dpi=300)

    plt.close(fig)


plot_vs_field(I_slow, I_slow_err, r'$I_s = A_s \tau_s$ [a.u.]', 'integral_slow')
plot_vs_field(I_fast, I_fast_err, r'$I_f = A_f \tau_f$ [a.u.]', 'integral_fast')
plot_vs_field(I_total, I_total_err, r'$I_s + I_f$ [a.u.]', 'integral_total')
plot_vs_field(f_fast, f_fast_err, r'$I_f / (I_s + I_f)$', 'fast_over_total')

# ----------------------------------------------------------------------
# integrais normalizadas em E = 0, e a diferenca entre as duas componentes
# ----------------------------------------------------------------------


def ratio_to_zero_field(v, verr):
    """v / v[E=0], com o erro relativo somado em quadratura.

    Os ajustes de campos diferentes sao independentes, mas TODOS dividem
    pelo mesmo v[0]: o erro de v[0] entra em cada ponto. Dai o termo
    (verr[0]/v[0])^2, que a divisao ingenua verr/v[0] deixava de fora.

    O ponto de E = 0 vale 1 por construcao, entao seu erro e zero -- a
    formula geral daria sqrt(2)*verr[0]/v[0] ali, que nao e flutuacao
    real, so o mesmo numero dividido por si mesmo."""

    r = v / v[0]

    r_err = np.abs(r) * np.sqrt((verr / v)**2 + (verr[0] / v[0])**2)
    r_err[0] = 0.0

    return r, r_err


r_slow,  r_slow_err  = ratio_to_zero_field(I_slow,  I_slow_err)
r_fast,  r_fast_err  = ratio_to_zero_field(I_fast,  I_fast_err)
r_total, r_total_err = ratio_to_zero_field(I_total, I_total_err)

# Quanto a componente rapida sobe a mais do que a lenta cai. Ja e uma
# diferenca de razoes normalizadas, entao I_dif(E=0) = 0 POR CONSTRUCAO --
# nao normalizar de novo por I_dif[0] (que e zero e mandava o erro para inf).
#
# cov(I_slow, I_fast) dentro de cada ajuste segue ignorada, mesma convencao
# usada em I_total_err.
I_dif = r_fast - r_slow
I_dif_err = np.sqrt(r_fast_err**2 + r_slow_err**2)

for name, r, r_err in [('integral_slow', r_slow, r_slow_err),
                       ('integral_fast', r_fast, r_fast_err),
                       ('integral_total', r_total, r_total_err)]:

    plot_vs_field(r, r_err, f'{name.replace("_", " ")} / E=0',
                  f'{name}_norm', hline=1.0)

plot_vs_field(I_dif, I_dif_err,
              r'$I_f/I_f^{0} - I_s/I_s^{0}$', 'integral_dif', hline=0.0)

# as duas componentes no mesmo eixo, normalizadas
fig, ax = plt.subplots()

ax.errorbar(efield, r_slow, yerr=r_slow_err,
            fmt='o', ms=4, label=r'$I_s$')
ax.errorbar(efield, r_fast, yerr=r_fast_err,
            fmt='s', ms=4, label=r'$I_f$')
ax.errorbar(efield, r_total, yerr=r_total_err,
            fmt='^', ms=4, label=r'$I_s + I_f$')
ax.errorbar(efield, I_dif, yerr=I_dif_err,
            fmt='v', ms=4, label=r'$I_f/I_f^{0} - I_s/I_s^{0}$')

ax.axhline(1.0, color='gray', ls=':', lw=0.8)
ax.axhline(0.0, color='gray', ls=':', lw=0.8)
ax.set_xlabel('Electric Field [kV/cm]')
ax.set_ylabel('Integral / (E = 0)')
ax.legend()

fig.savefig(f'{output_folder}/parameters/integrals_normalized.png', dpi=300)
plt.close(fig)

print('\n=== integrais (A*tau) ===')
print(f"{'kV/cm':>8} {'I_slow':>18} {'I_fast':>18} {'I_f/I_tot':>16}")
for i, f in enumerate(fields):
    print(f'{efield[i]:8.4f} '
          f'{I_slow[i]:9.4f} +- {I_slow_err[i]:.4f} '
          f'{I_fast[i]:9.4f} +- {I_fast_err[i]:.4f} '
          f'{f_fast[i]:8.4f} +- {f_fast_err[i]:.4f}')


# tau_slow vs campo -- Segreto 2021 (PRD 103, 043001), Eq. 14

tau_s = np.array([fit_results[f].values['tau_slow'] for f in fields])
tau_s_err = np.array([fit_results[f].errors['tau_slow'] for f in fields])


def tau_eff_segreto(E, alpha, beta, k_E):
    # Eq. 14 com beta/(1 + k_E/E) reescrito como beta*E/(E + k_E):
    # identico para E > 0 e finito em E = 0, onde tau_eff = 1/alpha
    return 1.0 / (alpha + beta * E / (E + k_E))


m_tau = Minuit(
    LeastSquares(efield, tau_s, tau_s_err, tau_eff_segreto),
    alpha=1.0 / tau_s[0], beta=0.2, k_E=0.075
)

m_tau.limits['alpha'] = (0.1, 5.0)
m_tau.limits['beta'] = (0, 5.0)
m_tau.limits['k_E'] = (0.001, 2.0)

m_tau.interactive()
m_tau.migrad()
m_tau.hesse()

ndof_tau = len(fields) - m_tau.nfit
chi2_tau = m_tau.fval / ndof_tau

alpha, beta, k_E = m_tau.values['alpha'], m_tau.values['beta'], m_tau.values['k_E']
tau_0 = 1.0 / alpha
tau_inf = 1.0 / (alpha + beta)

print('\n=== tau_slow(E): Segreto 2021, Eq. 14 ===')
print(m_tau)
print(f'tau_eff(0)   = {tau_0 * 1000:.1f} ns')
print(f'tau_eff(inf) = {tau_inf * 1000:.1f} ns   ({(tau_inf / tau_0 - 1) * 100:+.1f}%)')
print(f'k_E = {k_E:.4f} +- {m_tau.errors["k_E"]:.4f} kV/cm '
      f'[Segreto: 0.075 +- 0.015]')
print(f'chi2/ndof = {chi2_tau:.2f}')

E_smooth = np.linspace(0, efield.max() * 1.05, 400)

fig, ax = plt.subplots()

ax.errorbar(efield, tau_s * 1000, yerr=tau_s_err * 1000, fmt='o', ms=4)

ax.plot(E_smooth, tau_eff_segreto(E_smooth, alpha, beta, k_E) * 1000, 'r-', lw=1.2,
        label=(rf'$\tau_{{eff}}(0)$ = {tau_0 * 1000:.0f} ns' '\n'
               rf'$\tau_{{eff}}(\infty)$ = {tau_inf * 1000:.0f} ns' '\n'
               rf'$k_\mathcal{{E}}$ = {k_E:.3f} $\pm$ {m_tau.errors["k_E"]:.3f} kV/cm' '\n'
               rf'$\chi^2$/ndof = {chi2_tau:.1f}'))

ax.set_xlabel('Electric Field [kV/cm]')
ax.set_ylabel(r'$\tau_{slow}$ [ns]')
ax.legend()

fig.savefig(f'{output_folder}/parameters/tau_slow_segreto.png', dpi=300)
plt.close(fig)


# figura resumo: 4 paineis

fig, ax = plt.subplots(2, 2, figsize=(11, 8))

ax[0, 0].errorbar(efield, tau_s * 1000, yerr=tau_s_err * 1000, fmt='o', ms=4)
ax[0, 0].set_ylabel(r'$\tau_{slow}$ [ns]')
ax[0, 0].plot([], [], ' ',
              label=rf'$\tau_{{fast}}$ fixed at {TAU_FAST_FIXED * 1000:.0f} ns')
leg = ax[0, 0].legend(handlelength=0, handletextpad=0, frameon=False)
for txt in leg.get_texts():
    txt.set_color('darkorange')

ax[0, 1].errorbar(efield, f_fast, yerr=f_fast_err, fmt='o', ms=4)
ax[0, 1].set_ylabel(r'$F_{prompt} = I_f / (I_s + I_f)$')

ax[1, 0].errorbar(efield, I_fast, yerr=I_fast_err, fmt='o', ms=4)
ax[1, 0].set_ylabel(r'$I_f = A_f \tau_f$ [a.u.]')

ax[1, 1].errorbar(efield, I_slow, yerr=I_slow_err, fmt='o', ms=4)
ax[1, 1].set_ylabel(r'$I_s = A_s \tau_s$ [a.u.]')

for a in ax.flat:
    a.set_xlabel('Electric Field [kV/cm]')

fig.tight_layout()
fig.savefig(f'{output_folder}/parameters/summary_4panels.png', dpi=300)
plt.close(fig)


# ======================================================================
# dump dos parametros e incertezas vs campo, em txt
# ======================================================================
# Duas tabelas em {output_folder}/parameters/:
#
#   fit_params_vs_field.txt -- parametros crus do ajuste (valor e erro),
#                              um campo por linha, mais chi2/ndof e a
#                              correlacao A-tau de cada componente
#   integrals_vs_field.txt  -- derivados: I = A*tau de cada componente,
#                              total, F_prompt e as versoes normalizadas
#                              em E = 0
#
# rho(A, tau) vai junto de proposito: A e tau sao fortemente
# anticorrelacionados no ajuste, e sem esse numero nao da para propagar
# erro de nenhuma combinacao de I_s e I_f a partir do txt.
#
# Formato: metadados em linhas com '#', depois UMA linha de cabecalho sem
# '#' com os nomes das colunas. Le dos dois lados:
#     pd.read_csv(path, sep=r'\s+', comment='#')
#     np.genfromtxt(path, names=True, comments='#')

import datetime as _dt


def _write_table(path, meta, columns):
    """Tabela alinhada: columns e uma lista de (nome, sequencia)."""

    names = [c[0] for c in columns]
    cols = [list(c[1]) for c in columns]
    nrows = len(cols[0])

    def _fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return '1' if v else '0'
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return f'{float(v):.8g}'

    body = [[_fmt(col[i]) for col in cols] for i in range(nrows)]

    widths = [
        max([len(names[j])] + [row[j].__len__() for row in body])
        for j in range(len(names))
    ]

    with open(path, 'w') as fh:

        # cabecalho primeiro, metadados logo abaixo em linhas '#'. Nessa
        # ordem os dois leitores funcionam sem argumento extra:
        #   np.genfromtxt(path, names=True, comments='#')  -> le a 1a linha
        #   pd.read_csv(path, sep=r'\s+', comment='#')      -> pula os '#'
        fh.write('  '.join(names[j].rjust(widths[j])
                           for j in range(len(names))) + '\n')

        for line in meta:
            fh.write(f'# {line}\n')
        fh.write('#\n')

        for row in body:
            fh.write('  '.join(row[j].rjust(widths[j])
                               for j in range(len(names))) + '\n')

    print(f'[txt] {path}  ({nrows} campos, {len(names)} colunas)')


def _corr(m, name_A, name_tau):
    """rho(A, tau) do ajuste; nan se Hesse nao devolveu covariancia."""
    try:
        cov = m.covariance[name_A, name_tau]
        return cov / (m.errors[name_A] * m.errors[name_tau])
    except Exception:
        return np.nan


_fields_raw = list(fit_results.keys())

_tag = globals().get('CHANNEL', globals().get('MODULE', 'unknown'))


def _template_label():
    """CFG traz 'template' (um canal so) ou 'templates' (a media do
    modulo, dois canais). Aceita os dois para o bloco servir aos quatro
    scripts sem mudanca."""

    if 'template' in CFG:
        return CFG['template']

    return ' + '.join(CFG.get('templates', ['?']))


_meta_common = [
    f'gerado em     : {_dt.datetime.now():%Y-%m-%d %H:%M:%S}',
    f'script        : {os.path.basename(globals().get("__file__", "?"))}',
    f'canal/modulo  : {_tag}',
    f'template      : {_template_label()}',
    f'drift length  : {DRIFT_LENGTH} m   (E = HV / L)',
    f'janela do fit : {FIT_MIN} - {FIT_MAX} us',
    f'tau_fast      : FIXO em {TAU_FAST_FIXED} us',
    f'yerr          : YERR_MODE = {YERR_MODE}',
    'unidades      : campo em kV (HV) e kV/cm; tempos em us; '
    'amplitudes e integrais normalizadas pelo pico de E = 0',
]

# ---- tabela 1: parametros do ajuste ----------------------------------

_cols = [
    ('field_kV', _fields_raw),
    ('efield_kVcm', efield),
    ('run', [runs[f] for f in _fields_raw]),
]

for _p in params:
    _cols.append((_p, [fit_results[f].values[_p] for f in _fields_raw]))
    _cols.append((f'{_p}_err', [fit_results[f].errors[_p] for f in _fields_raw]))

_cols += [
    ('rho_Aslow_tauslow', [_corr(fit_results[f], 'Aslow', 'tau_slow')
                           for f in _fields_raw]),
    ('rho_Afast_taufast', [_corr(fit_results[f], 'Afast', 'tau_fast')
                           for f in _fields_raw]),
    ('chi2', [fit_stats[f]['chi2'] for f in _fields_raw]),
    ('ndof', [fit_stats[f]['ndof'] for f in _fields_raw]),
    ('chi2_ndof', [fit_stats[f]['chi2_ndof'] for f in _fields_raw]),
    ('valid', [fit_stats[f]['valid'] for f in _fields_raw]),
    ('accurate', [fit_stats[f]['accurate'] for f in _fields_raw]),
]

_write_table(
    f'{output_folder}/parameters/fit_params_vs_field.txt',
    _meta_common + [
        'valid/accurate: flags do Minuit (1 = ok). tau_fast fixo -> erro 0.',
    ],
    _cols,
)

# ---- tabela 2: integrais e derivados ---------------------------------

_write_table(
    f'{output_folder}/parameters/integrals_vs_field.txt',
    _meta_common + [
        'I_i = A_i * tau_i, erro com o termo cruzado cov(A, tau)',
        'r_i = I_i / I_i(E=0); o erro inclui o erro do denominador comum',
        'I_dif = r_fast - r_slow  (zero em E = 0 por construcao)',
    ],
    [
        ('field_kV', _fields_raw),
        ('efield_kVcm', efield),
        ('I_slow', I_slow),
        ('I_slow_err', I_slow_err),
        ('I_fast', I_fast),
        ('I_fast_err', I_fast_err),
        ('I_total', I_total),
        ('I_total_err', I_total_err),
        ('f_fast', f_fast),
        ('f_fast_err', f_fast_err),
        ('r_slow', r_slow),
        ('r_slow_err', r_slow_err),
        ('r_fast', r_fast),
        ('r_fast_err', r_fast_err),
        ('r_total', r_total),
        ('r_total_err', r_total_err),
        ('I_dif', I_dif),
        ('I_dif_err', I_dif_err),
    ],
)
