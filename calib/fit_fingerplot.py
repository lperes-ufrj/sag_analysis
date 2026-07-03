import uproot
import numpy as np
import matplotlib.pyplot as plt        
from iminuit import Minuit
from iminuit.cost import LeastSquares
import aux.functions as fc     
import os        

def fit_multigauss(f, run):
# iterate over all channels
    mean_spe = []
    std_spe = []

    # create folder if not exists

    os.makedirs(f"PLOTS/fingerplot/{run}/", exist_ok=True)
    os.makedirs(f"PLOTS/multigauss/{run}/", exist_ok=True)

    for i in range(len(f.keys())):
        integral = f[f.keys()[i]]["integral"].array(library="np")

        # Raw fingerplot
        x_bin = np.linspace(-200, 500, 120)
        plt.hist(integral, bins=x_bin, histtype='step', color='black')
        plt.title(f"{f.keys()[i]} - Fingerplot")
        plt.xlabel("Integral")
        plt.ylabel("Counts")
        plt.savefig(f"PLOTS/fingerplot/{run}/{f.keys()[i]}_fingerplot.png")
        plt.close()

        # Multigauss fingerplot
        counts, edges = np.histogram(integral, bins=x_bin)   
        centers = (edges[:-1] + edges[1:]) / 2
        y_err = np.sqrt(counts)
        y_err[y_err == 0] = 1                                  

        model = LeastSquares(centers, counts, y_err, fc.multi_gauss)

        initial_guesses = {
        "A0": 140  ,
        "q0": -30 ,
        "sigma0": 20,
        "gain":  100 ,
        "sigma1": 10,
        "sigma2": 20,
        "sigma3": 30,
        "sigma4": 40,
        "sigma5": 50,
        "A1": 90 ,
        "A2": 60,
        "A3": 20 ,
        "A4": 10,
        "A5": 5 ,
        }

        # minuit fit
        m = Minuit(model, **initial_guesses)
        #m.interactive()
        m.migrad()
        m.hesse()

        # Multigauss plot saved
        x_curve = np.linspace(centers[0], centers[-1], 500)
        y_curve = fc.multi_gauss(x_curve, *m.values)

        spe      = m.values["gain"]         
        spe_err  = m.errors["gain"]          
        ndf      = len(centers) - m.nfit    
        chi2_ndf = m.fval / ndf              

        label_fit = (f"multigauss fit\n"
                    f"SPE = {spe:.1f} ± {spe_err:.1f}\n"
                    f"$\\chi^2$/ndf = {chi2_ndf:.2f}")

        plt.errorbar(centers, counts, yerr=y_err, fmt='.', color='black', label='data')
        plt.plot(x_curve, y_curve, '-', color='red', label=label_fit)
        plt.title(f"{f.keys()[i]} - Multigauss")
        plt.xlabel("Integral")
        plt.ylabel("Counts")
        plt.legend()
        plt.savefig(f"PLOTS/multigauss/{run}/{f.keys()[i]}_multigauss.png")
        plt.close()

        # # crosstalk trial
        # integrals = np.array([fc.gaussian_integral(m.values[f"sigma{i}"]) for i in range(6)])
        # total_int_gauss = np.sum(integrals)

        # K_prob = integrals / total_int_gauss
        # x_ct = np.arange(6)

        # plt.plot(x_ct, K_prob, "*")
        # plt.savefig(f"PLOTS/ct/trial_ct_{f.keys()[i]}.png")
        # plt.close()

        print(f"{f.keys()[i]}: Multigauss Fitted")
        print(f"Nsamples = {len(integral)}")
        print(rf"SPE = {spe:.1f} +/- {spe_err:.1f}")
        print(rf"chi^2/ndf = {spe:.1f}")
        print(f"Fit Valid: {m.valid}")
        print()

        mean_spe.append(spe)
        std_spe.append(spe_err)

    return mean_spe, std_spe

