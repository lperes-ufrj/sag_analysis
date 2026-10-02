"""Integrais normalizadas do M8 vs campo: as duas componentes e a diferenca.

Le o txt gerado por fitting-bertolini-avg.py. Mesmo plot de
parameters/integrals_normalized.png, sem a curva da soma.
"""
import numpy as np
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle
import os

# caminhos ancorados no proprio script: funciona de qualquer diretorio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # sag_analysis/physics
REPO_DIR = os.path.dirname(SCRIPT_DIR)                    # sag_analysis
FIT_DIR = os.path.join(REPO_DIR, 'fitting')               # saidas do fitting-bertolini-avg.py

TXT = FIT_DIR + '/plots_bertolini_avg/M8/parameters/integrals_vs_field.txt'

d = np.genfromtxt(TXT, names=True, comments='#')

efield = d['efield_kVcm']

# I / I(E=0): colunas r_* do txt. O erro delas ja inclui o erro do
# denominador comum, entao nao da para normalizar aqui na mao.
Islow_m8, Islow_m8_err = d['r_slow'], d['r_slow_err']
Ifast_m8, Ifast_m8_err = d['r_fast'], d['r_fast_err']
Idif_m8, Idif_m8_err = d['I_dif'], d['I_dif_err']

fig, ax = plt.subplots()

ax.errorbar(efield, Islow_m8, yerr=Islow_m8_err,
            fmt='o', ms=4, label=r'$I_s$')
ax.errorbar(efield, Ifast_m8, yerr=Ifast_m8_err,
            fmt='s', ms=4, label=r'$I_f$')
ax.errorbar(efield, Idif_m8, yerr=Idif_m8_err,
            fmt='v', ms=4, label=r'$I_f/I_f^{0} - I_s/I_s^{0}$')

# canto superior direito: as curvas partem de 1 em E = 0, entao o canto
# esquerdo alto esta ocupado
dunestyle.Preliminary(x=0.97, y=0.92, align='right', ax=ax)

ax.axhline(1.0, color='gray', ls=':', lw=0.8)
ax.axhline(0.0, color='gray', ls=':', lw=0.8)
ax.set_xlabel('Electric Field [kV/cm]')
ax.set_ylabel('Integral / (E = 0)')
ax.legend()

plt.show()
