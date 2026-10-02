"""Ajuste da componente rapida extra (hipotese Ar++, Segreto 11/09/2025).

Nos observaveis normalizados em E = 0, com
    phi   = Phi(E)/Phi(0)      fator de excimeros
    theta = tau_q(E)/tau_0     sobrevivencia do tripleto
    g(E)  = E/(E + eps+)       componente extra, saturante
o modelo do Segreto da

    r_slow = phi * theta
    r_fast = phi + A*g(E)
    Delta  = r_fast - r_slow = phi*(1 - theta) + A*g(E)

com um unico parametro novo, A = alpha+ / (alpha_1 * Phi(0)): o tamanho da
componente extra relativa ao singlet em campo zero. N_i, alpha_1, alpha_3 e
a coleta de luz cancelam todos na normalizacao.

Duas identidades tiram o modelo de recombinacao do caminho:

    theta = tau_slow(E)/tau_slow(0)                      -- medido
    phi   = r_slow/theta = A_slow(E)/A_slow(0)           -- exato

Logo Delta_dado - Delta_modelo = r_fast - phi - A*g(E): o r_slow cancela dos
dois lados. O ajuste e sobre o RESIDUO r_fast - phi, que e o estimador da
componente extra sem nenhuma suposicao sobre R(E) -- e a curva do slide 19.
"""
import numpy as np
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle
from iminuit import Minuit
from iminuit.cost import LeastSquares
import os

# caminhos ancorados no proprio script: funciona de qualquer diretorio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # sag_analysis/physics
REPO_DIR = os.path.dirname(SCRIPT_DIR)                    # sag_analysis
FIT_DIR = os.path.join(REPO_DIR, 'fitting')               # saidas do fitting-bertolini-avg.py

DIR = FIT_DIR + '/plots_bertolini_avg/M8/parameters/'

EPS_SEGRETO = 0.003   # kV/cm -- eps+ da apresentacao (satura em ~40 V/cm)

par = np.genfromtxt(DIR + 'fit_params_vs_field.txt', names=True, comments='#')
ing = np.genfromtxt(DIR + 'integrals_vs_field.txt', names=True, comments='#')

E = ing['efield_kVcm']

phi = par['Aslow'] / par['Aslow'][0]
theta = par['tau_slow'] / par['tau_slow'][0]

Islow_m8, Islow_m8_err = ing['r_slow'], ing['r_slow_err']
Ifast_m8, Ifast_m8_err = ing['r_fast'], ing['r_fast_err']
Idif_m8, Idif_m8_err = ing['I_dif'], ing['I_dif_err']

# Erro ponto-a-ponto do residuo: so o numerador de cada razao. O ponto de
# E = 0 e um denominador COMUM a todos os pontos -- e uma escala global,
# nao flutuacao ponto-a-ponto, entao entra como sistematica e nao aqui.
# (Fica ignorada a covariancia entre A_slow e I_fast dentro do mesmo
#  ajuste, que o txt nao carrega.)
resid = Ifast_m8 - phi
resid_err = np.hypot(ing['I_fast_err'] / ing['I_fast'][0],
                     par['Aslow_err'] / par['Aslow'][0])

fit = E > 0        # E = 0 e zero por construcao, com erro zero: fora do fit


def extra(E, A, eps):
    """Componente extra normalizada: A * E/(E + eps+)."""
    return A * E / (E + eps)


cost = LeastSquares(E[fit], resid[fit], resid_err[fit], extra)

m = Minuit(cost, A=0.3, eps=0.2)
m.limits['A'] = (0, 2)
m.limits['eps'] = (1e-4, 5)
m.migrad()
m.hesse()

ndof = fit.sum() - m.nfit
print(m)
print(f'\nA     = {m.values["A"]:.4f} +- {m.errors["A"]:.4f}')
print(f'eps+  = {m.values["eps"]*1000:.0f} +- {m.errors["eps"]*1000:.0f} V/cm')
print(f'chi2/ndof = {m.fval/ndof:.2f}')

# mesma forma, com o eps+ da apresentacao fixo -- so para comparar
m_seg = Minuit(cost, A=0.2, eps=EPS_SEGRETO)
m_seg.limits['A'] = (0, 2)
m_seg.fixed['eps'] = True
m_seg.migrad()
m_seg.hesse()

ndof_seg = fit.sum() - m_seg.nfit
print(f'\ncom eps+ = {EPS_SEGRETO*1000:.0f} V/cm fixo (Segreto):')
print(f'A = {m_seg.values["A"]:.4f} +- {m_seg.errors["A"]:.4f}   '
      f'chi2/ndof = {m_seg.fval/ndof_seg:.2f}')
print(f'delta chi2 = {m_seg.fval - m.fval:.1f} para 1 grau de liberdade')

# --- plots ---------------------------------------------------------------
# Em cima: o que se quer descrever (Delta). Embaixo: o que de fato entra no
# ajuste (o residuo). A curva de Delta so existe nos pontos medidos, porque
# phi e theta sao dados e nao funcoes.

E_smooth = np.linspace(0, E.max() * 1.02, 400)

fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

ax[0].errorbar(E, Idif_m8, yerr=Idif_m8_err, fmt='v', ms=4, label=r'$\Delta$ (dado)')
ax[0].plot(E, phi * (1 - theta) + extra(E, *m.values), '-', lw=1.2,
           label=r'$\varphi(1-\theta) + A\,g$, ajuste')
ax[0].plot(E, phi * (1 - theta) + extra(E, m_seg.values['A'], EPS_SEGRETO),
           '--', lw=1.2, label=rf'idem, $\varepsilon^+$ = {EPS_SEGRETO*1000:.0f} V/cm')
ax[0].axhline(0.0, color='gray', ls=':', lw=0.8)
ax[0].set_ylabel(r'$I_f/I_f^{0} - I_s/I_s^{0}$')
# as curvas sobem de 0: canto superior esquerdo livre, legenda embaixo
dunestyle.Preliminary(ax=ax[0])
ax[0].legend(loc='lower right')

ax[1].errorbar(E, resid, yerr=resid_err, fmt='o', ms=4,
               label=r'$I_f/I_f^{0} - A_s/A_s^{0}$ (ajustado)')
ax[1].plot(E_smooth, extra(E_smooth, *m.values), '-', lw=1.2,
           label=(rf'$A$ = {m.values["A"]:.3f} $\pm$ {m.errors["A"]:.3f}' '\n'
                  rf'$\varepsilon^+$ = {m.values["eps"]*1000:.0f} $\pm$ '
                  rf'{m.errors["eps"]*1000:.0f} V/cm' '\n'
                  rf'$\chi^2$/ndof = {m.fval/ndof:.2f}'))
ax[1].plot(E_smooth, extra(E_smooth, m_seg.values['A'], EPS_SEGRETO), '--', lw=1.2,
           label=(rf'$\varepsilon^+$ = {EPS_SEGRETO*1000:.0f} V/cm fixo' '\n'
                  rf'$\chi^2$/ndof = {m_seg.fval/ndof_seg:.2f}'))
ax[1].axhline(0.0, color='gray', ls=':', lw=0.8)
ax[1].set_ylabel(r'componente extra  $A\,g(\mathcal{E})$')
ax[1].set_xlabel('Electric Field [kV/cm]')
dunestyle.Preliminary(ax=ax[1])
ax[1].legend(loc='lower right')

fig.tight_layout()
plt.show()
