from math import floor, log10
import numpy as np


def magnitude(x):
    return 3 - floor(log10(x))


def normalize(x_train, x_test):
    mu = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)

    x_train_normalized = (x_train - mu) / std
    x_test_normalized = (x_test - mu) / std

    return x_train_normalized, x_test_normalized


def rough_timedelta(seconds):
    seconds = abs(seconds)

    if seconds < 1:
        return 'less than 1 second'
    if seconds < 5:
        return '{:.1f} seconds'.format(seconds)

    if seconds < 120:
        return '{:.0f} seconds'.format(seconds)
    if seconds < 600:
        return '{:.0f} minutes {:.0f} seconds'.format(seconds // 60, seconds % 60)

    minutes = seconds / 60
    if minutes < 120:
        return '{:.0f} minutes'.format(minutes)
    if minutes < 600:
        return '{:.0f} hours {:.0f} minutes'.format(minutes // 60, minutes % 60)

    hours = minutes / 60
    if hours < 100:
        return '{:.0f} hours'.format(hours)
    if hours < 240:
        return '{:.0f} days {:.0f} hours'.format(hours // 24, hours % 24)

    days = hours / 24
    return '{:.0f} days'.format(days)
