import os
import time

import math
import numpy as np
import fastdtw
from scipy import stats
from scipy.signal import periodogram, welch
from biosppy.signals import ecg

import paths
import feature_utils
from read_data import read_balanced_data


def get_long_feature(long_signals):

    fs = 300
    feature_list = None
    long_features = list()

    step = 0
    for long_signal in long_signals:
        long_feature = dict()

        # extract features from the raw signals
        long_feature.update(long_basic_stat(long_signal))
        long_feature.update(long_zero_crossing(long_signal, threshold=0))
        long_feature.update(long_fft_band_power(long_signal, fs=fs))
        long_feature.update(long_fft_power(long_signal, fs=fs))
        long_feature.update(long_fft_band_power_shannon_entropy(long_signal, fs=fs))
        long_feature.update(long_snr(long_signal, fs=fs))

        # process a raw ECG signal and extract relevant signal features
        _, filtered, rpeaks, _, hbt, _, hr = ecg.ecg(signal=long_signal, sampling_rate=fs, show=False)

        # extract features from the heartbeat templates and r-peak info
        long_feature.update(short_wave_fft(hbt, fs=fs))
        long_feature.update(heart_rate_basic_stat(hr))
        long_feature.update(heart_beat_basic_stat(hbt, fs=fs))
        long_feature.update(long_medical(hbt, rpeaks, fs=fs))
        long_feature.update(filtered_long_r_peak(filtered, rpeaks))
        long_feature.update(dtw_distance_matrix(hbt))

        long_features.append(list(long_feature.values()))
        feature_list = long_feature.keys()

        step += 1
        if step % 1000 == 0:
            print('Step:', step, '/', len(long_signals))

    print('Long feature extraction is completed.')
    return feature_list, long_features


def long_basic_stat(long_signal):
    """
    Extract basic statistical features
    Refer to: https://github.com/hsd1503/ENCASE
    """
    feature = dict()

    feature['LongBasicStat_Range'] = max(long_signal) - min(long_signal)
    feature['LongBasicStat_Var'] = np.var(long_signal)
    feature['LongBasicStat_Skewness'] = stats.skew(long_signal)
    feature['LongBasicStat_Kurtosis'] = stats.kurtosis(long_signal)
    feature['LongBasicStat_Median'] = np.median(long_signal)

    feature['LongBasicStat_p1'] = np.percentile(long_signal, 1)
    feature['LongBasicStat_p5'] = np.percentile(long_signal, 5)
    feature['LongBasicStat_p10'] = np.percentile(long_signal, 10)
    feature['LongBasicStat_p25'] = np.percentile(long_signal, 25)
    feature['LongBasicStat_p75'] = np.percentile(long_signal, 75)
    feature['LongBasicStat_p90'] = np.percentile(long_signal, 90)
    feature['LongBasicStat_p95'] = np.percentile(long_signal, 95)
    feature['LongBasicStat_p99'] = np.percentile(long_signal, 99)

    feature['LongBasicStat_Range-99-1'] = feature['LongBasicStat_p99'] - feature['LongBasicStat_p1']
    feature['LongBasicStat_Range-95-5'] = feature['LongBasicStat_p95'] - feature['LongBasicStat_p5']
    feature['LongBasicStat_Range-90-10'] = feature['LongBasicStat_p90'] - feature['LongBasicStat_p10']
    feature['LongBasicStat_Range-75-25'] = feature['LongBasicStat_p75'] - feature['LongBasicStat_p25']

    return feature


def long_zero_crossing(long_signal, threshold=0):
    """
    Extract the number of times the signal crosses zero
    Refer to: https://github.com/hsd1503/ENCASE
    """
    feature = dict()

    count = 0
    for i in range(len(long_signal)-1):
        if (long_signal[i] - threshold) * (long_signal[i+1] - threshold) < 0:
            count += 1
        if long_signal[i] == threshold and \
                (long_signal[i-1] - threshold) * (long_signal[i+1] - threshold) < 0:
            count += 1

    feature['Long_ZeroCrossing'] = count

    return feature


def long_fft_band_power(long_signal, fs=300):
    """
    Extract the power spectral density of different partition of the signal
    Refer to: https://github.com/hsd1503/ENCASE (Method 1)
    """
    feature = dict()

    # Method 1
    length = len(long_signal)
    partition = [0, 1.5, 4, 8, 20, 100, fs / 2]

    _, psd = periodogram(long_signal, fs)  # f-sample frequencies; psd-Power spectral density
    partition = [int(x * length / fs) for x in partition]
    power = [sum(psd[partition[x]:partition[x + 1]]) for x in range(len(partition) - 1)]

    for i in range(len(power)):  # num of features: 6
        feature['LongFFTBandPower_' + str(i + 1)] = power[i]

    # # Method 2
    # _, psd = welch(long_signal, fs=fs)
    # psd = list(psd)
    #
    # for i in range(len(psd)):  # num of features: 129
    #     feature['LongFFTBandPower_' + str(i + 1)] = psd[i]

    return feature


def long_fft_power(long_signal, fs=300):
    """
    Refer to: https://github.com/hsd1503/ENCASE
    """
    feature = dict()

    ecg_fs_range = (0, 50)
    band_size = 5

    fxx, pxx = welch(long_signal, fs=fs)
    for i in range((ecg_fs_range[1] - ecg_fs_range[0]) // 5):
        fs_min = i * band_size
        fs_max = fs_min + band_size
        indices = np.logical_and(fxx >= fs_min, fxx < fs_max)
        bp = np.sum(pxx[indices])
        feature["LongFFTBandPower_Power-" + str(fs_min) + "-" + str(fs_max)] = bp

    return feature


def long_fft_band_power_shannon_entropy(long_signal, fs=300):
    """
    Extract entropy of power of each freq band
    Refer to: https://github.com/hsd1503/ENCASE
    """
    feature = dict()

    fs = fs
    length = len(long_signal)
    partition = [0, 1.5, 4, 8, 20, 100, fs/2]

    f, psd = periodogram(long_signal, fs)
    partition = [int(x * length / fs) for x in partition]
    p = [sum(psd[partition[x]:partition[x+1]]) for x in range(len(partition)-1)]
    prob = [x / sum(p) for x in p]
    entropy = sum([- x * math.log(x) for x in prob])

    feature['LongFFTBandPower_ShannonEntropy'] = entropy

    return feature


def long_snr(long_signal, fs=300):
    """
    Extract the signal-noise ratio of the signal
    Refer to: https://github.com/hsd1503/ENCASE
    """
    feature = dict()

    signal_power = 0
    noise_power = 0
    psd = periodogram(long_signal, fs=fs)

    for i in range(len(psd[0])):
        if psd[0][i] < 5:
            signal_power += psd[1][i]
        else:
            noise_power += psd[1][i]

    feature['LongSNR'] = signal_power / noise_power
    return feature


def short_wave_fft(hbt, fs=300):
    """
    Discrete Fourier transform of P/R/T wave
    """
    feature = dict()

    median_hbt = feature_utils.median_heartbeat(hbt, fs=fs)
    pff = feature_utils.extract_fft(median_hbt[:int(0.13 * fs)])
    rff = feature_utils.extract_fft(median_hbt[int(0.13 * fs):int(0.27 * fs)])
    tff = feature_utils.extract_fft(median_hbt[int(0.27 * fs):])

    for i, v in enumerate(pff[:10]):
        feature['PWaveFFT_' + str(i)] = v
    for i, v in enumerate(rff[:10]):
        feature['RWaveFFT_' + str(i)] = v
    for i, v in enumerate(tff[:20]):
        feature['TWaveFFT_' + str(i)] = v

    return feature


def heart_rate_basic_stat(hr):
    """
    Extract the heart rate statistical features
    """
    feature = {
        'HRBasicStat_Max': 0,
        'HRBasicStat_Min': 0,
        'HRBasicStat_Mean': 0,
        'HRBasicStat_Median': 0,
        'HRBasicStat_Mode': 0,
        'HRBasicStat_Std': 0,
        'HRBasicStat_Skewness': 0,
        'HRBasicStat_Kurtosis': 0,
        'HRBasicStat_Range': 0,
        'HRBasicStat_Count': 0
    }

    if len(hr) > 0:
        feature['HRBasicStat_Max'] = np.amax(hr)
        feature['HRBasicStat_Min'] = np.amin(hr)
        feature['HRBasicStat_Mean'] = np.mean(hr)
        feature['HRBasicStat_Median'] = np.median(hr)
        feature['HRBasicStat_Mode'] = stats.mode(hr, axis=None)[0][0]
        feature['HRBasicStat_Std'] = np.std(hr)
        feature['HRBasicStat_Skewness'] = stats.skew(hr)
        feature['HRBasicStat_Kurtosis'] = stats.kurtosis(hr)
        feature['HRBasicStat_Range'] = np.amax(hr) - np.amin(hr)
        feature['HRBasicStat_Count'] = len(hr)

    return feature


def heart_beat_basic_stat(hbt, fs=300):
    """
    Extract the heart beat statistical features
    """
    feature = dict()

    means = feature_utils.median_heartbeat(hbt, fs=fs)
    feature['MedianHeartBeat_p5'] = np.percentile(means, 5)
    feature['MedianHeartBeat_p25'] = np.percentile(means, 25)
    feature['MedianHeartBeat_p75'] = np.percentile(means, 75)
    feature['MedianHeartBeat_p95'] = np.percentile(means, 95)

    return feature


def long_medical(hbt, rpeaks, fs=300):
    """
    Extract medical features
    """

    feature = dict()

    means = feature_utils.median_heartbeat(hbt, fs=fs)
    diff_rpeak = np.median(np.diff(rpeaks))

    if diff_rpeak < 280:
        tmp1 = 100 - int(diff_rpeak / 3)
        tmp2 = 100 + int(2 * diff_rpeak / 3)
        for i in range(len(means)):
            if i < tmp1:
                means[i] = 0
            elif i > tmp2:
                means[i] = 0
        freq_cof = diff_rpeak / 300 * fs
    else:
        freq_cof = fs

    r_pos = int(0.2 * fs)
    PQ = means[r_pos - int(0.2 * freq_cof):r_pos - int(0.05 * freq_cof)]
    ST = means[r_pos + int(0.05 * freq_cof):r_pos + int(0.4 * freq_cof)]
    QR = means[r_pos - int(0.07 * freq_cof):r_pos]
    RS = means[r_pos:r_pos + int(0.07 * freq_cof)]

    q_pos = r_pos - len(QR) + np.argmin(QR)
    s_pos = r_pos + np.argmin(RS)
    p_pos = np.argmax(PQ)
    t_pos = np.argmax(ST)

    feature['Medical_QRS-Length'] = s_pos - q_pos
    feature['Medical_PR-Interval'] = r_pos - p_pos
    feature['Medical_ST-Interval'] = t_pos

    feature['Medical_P-Max'] = PQ[p_pos]
    feature['Medical_T-Max'] = ST[t_pos]
    feature['Medical_P-to-Q'] = PQ[p_pos] - means[q_pos]
    feature['Medical_T-to-S'] = ST[t_pos] - means[s_pos]

    t_wave = ST[max(0, t_pos - int(0.08 * freq_cof)):min(len(ST), t_pos + int(0.08 * freq_cof))]
    p_wave = PQ[max(0, p_pos - int(0.06 * freq_cof)):min(len(PQ), p_pos + int(0.06 * freq_cof))]
    feature['Medical_P-Skewness'] = stats.skew(p_wave)
    feature['Medical_P-Kurtosis'] = stats.kurtosis(p_wave)
    feature['Medical_T-Skewness'] = stats.skew(t_wave)
    feature['Medical_T-Kurtosis'] = stats.kurtosis(t_wave)

    QRS = means[q_pos:s_pos]
    qrs_min = abs(min(QRS))
    qrs_max = abs(max(QRS))
    feature['Medical_QS-Diff'] = abs(means[s_pos] - means[q_pos])
    feature['Medical_QRS-Kurtosis'] = stats.kurtosis(QRS)
    feature['Medical_QRS-Skewness'] = stats.skew(QRS)
    feature['Medical_QRS-MinMax'] = qrs_max - qrs_min

    feature['Medical_Activity'] = feature_utils.calc_activity(means)
    feature['Medical_Mobility'] = feature_utils.calc_mobility(means)
    feature['Medical_Complexity'] = feature_utils.calc_complexity(means)

    return feature


def filtered_long_r_peak(filtered_signal, rpeaks):
    """
    Extract the r-peak features from the filtered signal
    """
    feature = dict()

    r_vals = [filtered_signal[i] for i in rpeaks]
    rr_intervals = np.diff(rpeaks)
    mean_rr_intervals = np.mean(rr_intervals)
    filtered_count = sum([1 if i < 0.5 * mean_rr_intervals else 0 for i in rr_intervals])

    total = len(r_vals) if len(r_vals) > 0 else 1

    feature['FilteredR_Beats-to-Length'] = len(rpeaks) / len(filtered_signal)
    feature['FilteredR_R-Mean'] = np.mean(r_vals)
    feature['FilteredR_R-Std'] = np.std(r_vals)
    feature['FilteredR_Count'] = filtered_count
    feature['FilteredR_Rel-Filtered-R'] = filtered_count / total

    return feature


def dtw_distance_matrix(hbt):
    """
    Extract the features from the approximate (Euclidean) distances between 2 time series
    """
    feature = dict()

    num_hbt = len(hbt)
    tmp_distance_dist = [fastdtw.fastdtw(hbt[i], hbt[i+1], dist=2)[0] for i in range(num_hbt-1)]

    feature["DTW_Max"] = max(tmp_distance_dist)
    feature["DTW_Min"] = min(tmp_distance_dist)
    feature["DTW_Range"] = max(tmp_distance_dist) - min(tmp_distance_dist)
    feature["DTW_Var"] = np.var(tmp_distance_dist)
    feature["DTW_Skewness"] = stats.skew(tmp_distance_dist)
    feature["DTW_Kurtosis"] = stats.kurtosis(tmp_distance_dist)
    feature["DTW_Median"] = np.median(tmp_distance_dist)

    return feature


# test
if __name__ == '__main__':

    long_signals, _, _ = read_balanced_data(folder=paths.parsed_train_folder, datatype='long')

    b_t = time.time()
    feature_list, long_features = get_long_feature(long_signals)
    e_t = time.time()

    print("Time to extract long features from {0} long signals: {1}".format(
        len(long_signals), (e_t - b_t)))
    print("Num of features: ", len(long_features[0]))
    print(feature_list)
    print(long_features[0])



