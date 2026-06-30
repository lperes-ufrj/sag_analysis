import uproot
import numpy as np
import matplotlib.pyplot as plt        
from iminuit import Minuit
from iminuit.cost import LeastSquares
import aux.functions as fc   
import fingerplot_fit as fg_fit   

# read selected wfs from read_waveforms
f = uproot.open(".roots/metrics_select.root")

# fingerplot multigauss fit
fg_fit.fit_multigauss(f)