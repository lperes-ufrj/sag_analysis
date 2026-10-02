"""Ajuste da componente rapida extra (hipotese Ar++, Segreto 11/09/2025),
por modulo ou para a MEDIA de varios modulos -- generaliza fit_diff.py (que
so fazia M8) e adiciona um segundo estudo: dá pra tambem descrever o
f-prompt com o mesmo modelo?

    python fit_diff_module.py M5 M8          # um fit por modulo (default)
    python fit_diff_module.py M5 M8 --mean   # um fit so, na media de M5 e M8

Nos observaveis normalizados em E = 0 (que aqui coincide com o PICO, ja que
Aslow, tau_slow, I_fast e I_slow caem monotonicamente com o campo nos 3
modulos -- ver nota grande mais abaixo sobre normalizacao), com

    phi   = Phi(E)/Phi(0)      fator de excimeros = A_slow(E)/A_slow(0)
    theta = tau_q(E)/tau_0     sobrevivencia do tripleto
    g(E)  = E/(E + eps+)       componente extra, saturante

o modelo do Segreto da

    r_slow = phi * theta
    r_fast = phi + A*g(E)
    Delta  = r_fast - r_slow = phi*(1 - theta) + A*g(E)

com um unico parametro novo, A = alpha+ / (alpha_1 * Phi(0)).

Os chutes iniciais dos fits vem da bibliografia quando ela da um numero:
eps+ comeca em EPS_SEGRETO (leitura do slide 19 -- ver comentario abaixo).
A nao tem valor citado em lugar nenhum, fica num chute generico.

--- --mean: como funciona -------------------------------------------------
phi, theta, resid, I_dif e f_fast de cada modulo sao medias ponto-a-ponto
(mesma grade de campo nos 3 modulos); a incerteza e o desvio entre modulos
dividido por sqrt(n) (erro padrao da media, mesma convencao do
plot_module_comparison.py). Para f_prompt tambem precisa da razao
I_slow(0)/I_fast(0) de cada modulo -- essa tambem entra na media, o que
evita ter que carregar I_fast(0)/I_slow(0) em escala absoluta (que difere
muito entre modulos).

--- f-prompt: e possivel? -------------------------------------------------
f_prompt = I_fast/(I_fast+I_slow) e so uma reparametrizacao das MESMAS duas
componentes que Delta ja usa -- ele nao introduz nenhuma fisica nova, entao
se o modelo acima descreve I_fast e I_slow separadamente (via r_fast, r_slow
escalados pela razao I_slow(0)/I_fast(0)), ele automaticamente da uma
previsao para f_prompt tambem, SEM nenhum parametro extra:

    f_prompt_model(E) = 1 / (1 + (I_slow(0)/I_fast(0)) * r_slow(E)/r_fast(E))

Isso e testado de duas formas: (1) so reaproveitando A, eps+ do ajuste do
Delta (checagem de consistencia, zero parametros novos); (2) um ajuste
INDEPENDENTE de f_prompt com A, eps+ livres, direto nos dados de f_fast, pra
ver se convergem para valores parecidos -- essa e a resposta real de "da pra
ajustar f-prompt": sim, e o mesmo aparato, e o teste de consistencia entre
os dois ajustes (via Delta e via f_prompt direto) e um teste do modelo.

--- normalizacao: afeta esse ajuste? --------------------------------------
Nao, pelo menos no scan atual: phi e theta aqui sao normalizados pelo ponto
E = 0 (ratio_to_zero_field, igual ao I_dif de fitting-bertolini-avg.py), e
NAO pelo "pico" generico que plot_module_comparison.py usa (argmax de
|valor|). Mas como Aslow(E), tau_slow(E), I_fast(E) e I_slow(E) caem
monotonicamente com o campo nos 3 modulos, o pico de cada uma delas SEMPRE
cai em E = 0 -- entao "normalizar pelo pico" e "normalizar por E = 0" dao
exatamente o mesmo numero aqui. A unica grandeza onde isso faria diferenca
e f_prompt, que CRESCE com o campo (pico no maior E, nao em E = 0) -- mas
f_prompt nao entra normalizado em lugar nenhum deste ajuste, entao tambem
nao ha efeito.
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
BASE = FIT_DIR + '/plots_bertolini_avg/{mod}/parameters/'
OUTDIR = os.path.join(SCRIPT_DIR, 'plots/fit_diff') + '/'

EPS_SEGRETO = 0.003   # kV/cm -- eps+ da apresentacao (satura em ~40 V/cm,
                       # ver slides 10 e 19: "it starts decaying after ~50 V/cm").
                       # Usado so como CHUTE INICIAL do fit livre -- nao tem
                       # mais um fit separado com eps+ fixo nesse valor
                       # (tirado: com eps+ = 3 V/cm a curva satura quase
                       # imediatamente e nao parece com o dado nenhum pouco).


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('modules', nargs='*', metavar='MODULE',
                    help=f'modulos a ajustar, entre {ALL_MODULES} (default: M5 M8)')
    p.add_argument('--mean', action='store_true',
                    help='em vez de um fit por modulo, faz UM fit na media dos modulos listados')
    args = p.parse_args()
    args.modules = args.modules or ['M5', 'M8']
    bad = [m for m in args.modules if m not in ALL_MODULES]
    if bad:
        p.error(f'modulo(s) invalido(s): {bad} -- escolha entre {ALL_MODULES}')
    return args.modules, args.mean


def add_watermarks(ax):
    dunestyle.Preliminary(x=0.05, y=0.08, ax=ax, fontsize=11)
    dunestyle.WIP(x=0.05, y=0.03, ax=ax, fontsize=11)


def extra(E, A, eps):
    """Componente extra normalizada: A * E/(E + eps+)."""
    return A * E / (E + eps)


def load(mod):
    base = BASE.format(mod=mod)
    par = np.genfromtxt(base + 'fit_params_vs_field.txt', names=True, comments='#')
    ing = np.genfromtxt(base + 'integrals_vs_field.txt', names=True, comments='#')

    # os dois txt devem vir do MESMO run de fitting-bertolini-avg.py (mesmos
    # campos, na mesma ordem). Se um dos dois ficou desatualizado -- por
    # exemplo o script foi interrompido entre escrever um e outro -- os
    # arrays saem de tamanhos diferentes e o erro so aparece la na frente,
    # dentro do fit, sem dizer o motivo real. Falha aqui, cedo e claro.
    if not np.array_equal(par['field_kV'], ing['field_kV']):
        raise ValueError(
            f'{mod}: fit_params_vs_field.txt ({len(par)} campos) e '
            f'integrals_vs_field.txt ({len(ing)} campos) nao batem -- '
            'rode fitting-bertolini-avg.py de novo para este modulo '
            '(os dois arquivos saem desse mesmo script, um logo apos o outro).')

    return par, ing


def module_quantities(mod):
    """Tudo que os dois fits precisam, para UM modulo."""
    par, ing = load(mod)

    E = ing['efield_kVcm']
    phi = par['Aslow'] / par['Aslow'][0]
    theta = par['tau_slow'] / par['tau_slow'][0]

    resid = ing['r_fast'] - phi
    resid_err = np.hypot(ing['I_fast_err'] / ing['I_fast'][0],
                         par['Aslow_err'] / par['Aslow'][0])

    return dict(
        E=E, phi=phi, theta=theta, resid=resid, resid_err=resid_err,
        I_dif=ing['I_dif'], I_dif_err=ing['I_dif_err'],
        f_fast=ing['f_fast'], f_fast_err=ing['f_fast_err'],
        ratio0=ing['I_slow'][0] / ing['I_fast'][0],
    )


def mean_quantities(mods):
    """Media ponto-a-ponto das quantidades de varios modulos, com incerteza
    = desvio entre modulos / sqrt(n) (erro padrao da media)."""
    qs = [module_quantities(m) for m in mods]

    E = qs[0]['E']
    if not all(np.allclose(q['E'], E) for q in qs[1:]):
        raise ValueError(f'{mods}: os modulos nao tem a mesma grade de campo')

    n = len(qs)

    def avg(key):
        stack = np.array([q[key] for q in qs])
        mean = stack.mean(axis=0)
        sem = (stack.std(axis=0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros_like(mean)
        return mean, sem

    phi, _ = avg('phi')
    theta, _ = avg('theta')
    resid, resid_err = avg('resid')
    I_dif, I_dif_err = avg('I_dif')
    f_fast, f_fast_err = avg('f_fast')
    ratio0 = np.mean([q['ratio0'] for q in qs])

    return dict(E=E, phi=phi, theta=theta, resid=resid, resid_err=resid_err,
                I_dif=I_dif, I_dif_err=I_dif_err, f_fast=f_fast, f_fast_err=f_fast_err,
                ratio0=ratio0)


def fit_idif(label, q):
    """Ajuste do residuo r_fast - phi = A*g(E) (equivalente a ajustar Delta,
    ver docstring)."""
    E, resid, resid_err = q['E'], q['resid'], q['resid_err']

    fit_mask = E > 0   # E = 0 e zero por construcao, com erro zero: fora do fit

    cost = LeastSquares(E[fit_mask], resid[fit_mask], resid_err[fit_mask], extra)

    m = Minuit(cost, A=0.3, eps=EPS_SEGRETO)
    m.limits['A'] = (0, 2)
    m.limits['eps'] = (1e-4, 5)
    m.interactive()
    plt.close('all')
    m.migrad()
    m.hesse()

    ndof = fit_mask.sum() - m.nfit

    print(f'\n=== {label}: ajuste de Delta (r_fast - r_slow), A e eps+ livres ===')
    print(m)
    print(f'A     = {m.values["A"]:.4f} +- {m.errors["A"]:.4f}')
    print(f'eps+  = {m.values["eps"]*1000:.0f} +- {m.errors["eps"]*1000:.0f} V/cm')
    print(f'chi2/ndof = {m.fval/ndof:.2f}')

    return dict(m=m, ndof=ndof, fit_mask=fit_mask)


def plot_idif(label, q, fit_out):
    E, phi, theta = q['E'], q['phi'], q['theta']
    resid, resid_err = q['resid'], q['resid_err']
    Idif, Idif_err = q['I_dif'], q['I_dif_err']
    m, ndof = fit_out['m'], fit_out['ndof']

    E_smooth = np.linspace(0, E.max() * 1.02, 400)

    fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    ax[0].errorbar(E, Idif, yerr=Idif_err, fmt='v', ms=4, label=r'$\Delta$ (dado)')
    ax[0].plot(E, phi * (1 - theta) + extra(E, *m.values), '-', lw=1.2,
               label=r'$\varphi(1-\theta) + A\,g$, ajuste')
    ax[0].axhline(0.0, color='gray', ls=':', lw=0.8)
    ax[0].set_ylabel(r'$I_f/I_f^{0} - I_s/I_s^{0}$')
    ax[0].set_title(label)
    add_watermarks(ax[0])
    ax[0].legend(loc='best')

    ax[1].errorbar(E, resid, yerr=resid_err, fmt='o', ms=4,
                   label=r'$I_f/I_f^{0} - A_s/A_s^{0}$ (ajustado)')
    ax[1].plot(E_smooth, extra(E_smooth, *m.values), '-', lw=1.2,
               label=(rf'$A$ = {m.values["A"]:.3f} $\pm$ {m.errors["A"]:.3f}' '\n'
                      rf'$\varepsilon^+$ = {m.values["eps"]*1000:.0f} $\pm$ '
                      rf'{m.errors["eps"]*1000:.0f} V/cm' '\n'
                      rf'$\chi^2$/ndof = {m.fval/ndof:.2f}'))
    ax[1].axhline(0.0, color='gray', ls=':', lw=0.8)
    ax[1].set_ylabel(r'componente extra  $A\,g(\mathcal{E})$')
    ax[1].set_xlabel('Electric Field [kV/cm]')
    add_watermarks(ax[1])
    ax[1].legend(loc='best')

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}idiff_fit_{label}.png', dpi=150)
    plt.close(fig)


def fit_fprompt(label, q, fit_out):
    """f_prompt derivado do MESMO r_fast, r_slow (0 parametros novos) vs um
    ajuste INDEPENDENTE de f_prompt com A, eps+ livres -- ver docstring."""
    E, phi, theta, ratio0 = q['E'], q['phi'], q['theta'], q['ratio0']
    f_fast, f_fast_err = q['f_fast'], q['f_fast_err']

    def fprompt_model(E, A, eps):
        r_fast = phi + extra(E, A, eps)
        r_slow = phi * theta
        return 1.0 / (1.0 + ratio0 * r_slow / r_fast)

    # previsao so com A, eps+ do ajuste do Delta -- checagem de consistencia
    fprompt_from_idif = fprompt_model(E, fit_out['m'].values['A'], fit_out['m'].values['eps'])
    chi2_from_idif = np.sum(((f_fast - fprompt_from_idif) / f_fast_err) ** 2)

    # ajuste independente, direto em f_fast
    cost = LeastSquares(E, f_fast, f_fast_err, fprompt_model)
    m_f = Minuit(cost, A=0.3, eps=EPS_SEGRETO)
    m_f.limits['A'] = (0, 2)
    m_f.limits['eps'] = (1e-4, 5)
    m_f.interactive()
    plt.close('all')
    m_f.migrad()
    m_f.hesse()

    ndof_f = len(E) - m_f.nfit

    print(f'\n=== {label}: f_prompt -- previsao vs ajuste direto ===')
    print(f'so com A, eps+ do ajuste de Delta: '
          f'A = {fit_out["m"].values["A"]:.4f}, eps+ = {fit_out["m"].values["eps"]*1000:.0f} V/cm'
          f'   chi2/ndof = {chi2_from_idif/len(E):.2f} (ndof aproximado = n pontos)')
    print(m_f)
    print(f'ajuste direto: A = {m_f.values["A"]:.4f} +- {m_f.errors["A"]:.4f}   '
          f'eps+ = {m_f.values["eps"]*1000:.0f} +- {m_f.errors["eps"]*1000:.0f} V/cm   '
          f'chi2/ndof = {m_f.fval/ndof_f:.2f}')

    E_smooth = np.linspace(0, E.max() * 1.02, 400)

    fig, ax = plt.subplots()
    ax.errorbar(E, f_fast, yerr=f_fast_err, fmt='o', ms=4, label=r'$f_{prompt}$ (dado)')

    phi_smooth = np.interp(E_smooth, E, phi)
    theta_smooth = np.interp(E_smooth, E, theta)

    def fprompt_smooth(A, eps):
        r_fast = phi_smooth + extra(E_smooth, A, eps)
        r_slow = phi_smooth * theta_smooth
        return 1.0 / (1.0 + ratio0 * r_slow / r_fast)

    ax.plot(E_smooth, fprompt_smooth(fit_out['m'].values['A'], fit_out['m'].values['eps']),
            '--', lw=1.2,
            label=(r'previsao (A, $\varepsilon^+$ do ajuste de $\Delta$)' '\n'
                   rf'$A$ = {fit_out["m"].values["A"]:.3f}, '
                   rf'$\varepsilon^+$ = {fit_out["m"].values["eps"]*1000:.0f} V/cm'))
    ax.plot(E_smooth, fprompt_smooth(m_f.values['A'], m_f.values['eps']),
            '-', lw=1.2,
            label=(r'ajuste direto em $f_{prompt}$' '\n'
                   rf'$A$ = {m_f.values["A"]:.3f} $\pm$ {m_f.errors["A"]:.3f}, '
                   rf'$\varepsilon^+$ = {m_f.values["eps"]*1000:.0f} $\pm$ '
                   rf'{m_f.errors["eps"]*1000:.0f} V/cm' '\n'
                   rf'$\chi^2$/ndof = {m_f.fval/ndof_f:.2f}'))

    ax.set_xlabel('Electric Field [kV/cm]')
    ax.set_ylabel(r'$f_{prompt}$')
    ax.set_title(label)
    add_watermarks(ax)
    ax.legend(loc='best', fontsize=8)

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}fprompt_fit_{label}.png', dpi=150)
    plt.close(fig)


def run(label, q):
    fit_out = fit_idif(label, q)
    plot_idif(label, q, fit_out)
    fit_fprompt(label, q, fit_out)


def main():
    modules, mean = parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    if mean:
        label = 'mean_' + '-'.join(modules)
        run(label, mean_quantities(modules))
    else:
        for mod in modules:
            run(mod, module_quantities(mod))

    print('\ndone ->', OUTDIR)


if __name__ == '__main__':
    main()
