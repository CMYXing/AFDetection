import numpy as np
from scipy.fftpack import rfft


def median_heartbeat(hb_templates, fs=300):

    if len(hb_templates) == 0:
        return np.zeros((int(0.6 * fs)), dtype=np.int32)

    median = [np.median(col) for col in hb_templates.T]

    dists = [np.sum(np.square(s - median)) for s in hb_templates]
    pmin = np.argmin(dists)

    median_hbt = hb_templates[pmin]

    r_pos = int(0.2 * fs)
    if median_hbt[r_pos] < 0:
        median_hbt *= -1

    return median_hbt  # shape: (180, 0)


def extract_fft(x):
    return rfft(x)[:len(x) // 2]


def calc_activity(epoch):
    """
    Calculate Hjorth activity over epoch
    """
    return np.nanvar(epoch, axis=0)


def calc_mobility(epoch):
    """
    Calculate the Hjorth mobility parameter over epoch
    """
    # Mobility
    # N.B. the sqrt of the variance is the standard deviation. So let's just get std(dy/dt) / std(y)
    return np.divide(np.nanstd(np.diff(epoch, axis=0)), np.nanstd(epoch, axis=0))


def calc_complexity(epoch):
    """
    Calculate Hjorth complexity over epoch
    """
    return np.divide(calc_mobility(np.diff(epoch, axis=0)), calc_mobility(epoch))
