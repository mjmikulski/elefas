from math import floor, log10
import numpy as np

def magnitude(x) -> int:
    return 3 - floor(log10(x))

def normalize(x_train, x_test):
    mu = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)

    x_train_normalized = (x_train - mu) / std
    x_test_normalized = (x_test - mu) / std

    return x_train_normalized, x_test_normalized