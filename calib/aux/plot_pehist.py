import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
from matplotlib.colors import LogNorm
from calib.aux import functions as fc

def plot_PE_2dhist(f, mean_spe):
    keys = fc.latest_cycle_keys(f)

    for i, key in enumerate(keys):
        wfs = f[key]["wf_nobaseline"].array()
        wfs_pe = wfs / mean_spe[i]
        Y_axe = ak.to_numpy(ak.flatten(wfs_pe))

        n_waveforms = len(wfs)
        pontos = len(wfs[0])

        x_axe = np.arange(pontos)
        X_axe = np.tile(x_axe, n_waveforms)

        plt.figure(figsize=(10,6))
        h = plt.hist2d(X_axe, Y_axe, bins=[pontos, 150], cmap='viridis', norm=LogNorm())
        
        plt.colorbar(h[3], label='Counts')  
        plt.title(f"{key}: PE Histogram")
        plt.xlabel("Samples")
        plt.ylabel("PE")
        plt.show()