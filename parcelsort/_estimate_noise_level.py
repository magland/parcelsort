import numpy as np

def estimate_noise_level(X: np.array):
    return np.median(np.abs(X.squeeze())) / 0.6745  # median absolute deviation (MAD) estimate of stdev