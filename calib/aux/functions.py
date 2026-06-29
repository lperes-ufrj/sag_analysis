import os, pickle, re, glob
import numpy as np
import matplotlib.pyplot as plt
import math

def multi_gauss(q, A0, q0, sigma0, gain,
                        sigma1, sigma2, sigma3, sigma4, sigma5,
                        A1, A2, A3, A4, A5):
    y      = A0 * np.exp(-0.5 * ((q - q0) / sigma0)**2)
    sigmas = [sigma1, sigma2, sigma3, sigma4, sigma5]
    amps   = [A1, A2, A3, A4, A5]
    for n_idx, (An, sig) in enumerate(zip(amps, sigmas)):
        y += An * np.exp(-0.5 * ((q - (q0 + (n_idx + 1) * gain)) / sig)**2)
    return y

def prob_crosstalk_func(k_arr, L, p):
    y_raw = []
    for k_val in k_arr:
        k = int(k_val)
        if k == 0:
            y_raw.append(np.exp(-L))
        else:
            summ = 0.0
            for i in range(1, k + 1):
                num  = math.factorial(k - 1)
                den  = math.factorial(i) * math.factorial(i - 1) * math.factorial(k - i)
                B_ik = num / den
                term = B_ik * ((L * (1 - p))**i) * (p**(k - i))
                summ += term
            y_raw.append(np.exp(-L) * summ)
    return np.array(y_raw)

