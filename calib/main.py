import uproot
import numpy as np
import matplotlib.pyplot as plt        
from iminuit import Minuit
from iminuit.cost import LeastSquares
import aux.functions as fc   
import fit_fingerplot as fg_fit   
import plot_pehist as pl_pe

run = str(input("Enter run number: "))

# read selected wfs from read_waveforms
filename = ".roots/" + "metrics_select/run" + run + "_metrics_select.root"
f = uproot.open(filename)

# fingerplot multigauss fit
mean_spe, std_spe = fg_fit.fit_multigauss(f, run)

# # plot channel PE 2d histogram\
# pl_pe.plot_PE_2dhist(f, mean_spe)