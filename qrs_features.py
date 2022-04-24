import os
import time

import numpy as np
from scipy import stats

import paths
from read_data import read_qrs_data


def get_qrs_feature(qrs_infos):

    feature_list = None
    qrs_features = list()

    step = 0
    for qrs_info in qrs_infos:
        qrs_info = qrs_info[1:-1]
        qrs_feature = dict()

        qrs_feature.update(rri_basic_stat(qrs_info))
        qrs_feature.update(rri_basic_stat_point_median(qrs_info))
        qrs_feature.update(rri_basic_stat_deltaRR(qrs_info))
        qrs_feature.update(cdf(qrs_info))
        qrs_feature.update(coeff_of_variation(qrs_info))
        qrs_feature.update(median_absolute_deviation(qrs_info))
        qrs_feature.update(hrv_analysis(qrs_info))

        qrs_features.append(list(qrs_feature.values()))
        feature_list = qrs_feature.keys()

        step += 1
        if step % 1000 == 0:
            print('Step:', step, '/', len(qrs_infos))

    print('QRS feature extraction is completed.')
    return feature_list, qrs_features


def rri_basic_stat(qrs_info):
    """
    Extract basic statistical features of rr-interval
    """
    feature = {
        'RRIBasicStat_Mean': 0.0,
        'RRIBasicStat_Hr': 0.0,
        'RRIBasicStat_NumPeaks': 0.0,
        'RRIBasicStat_Range': 0.0,
        'RRIBasicStat_Var': 0.0,
        'RRIBasicStat_Std': 0.0,
        'RRIBasicStat_Skewness': 0.0,
        'RRIBasicStat_Kurtosis': 0.0,
        'RRIBasicStat_Median': 0.0,
        'RRIBasicStat_Min': 0.0,
        'RRIBasicStat_p5': 0.0,
        'RRIBasicStat_p25': 0.0,
        'RRIBasicStat_p75': 0.0,
        'RRIBasicStat_p95': 0.0,
        'RRIBasicStat_Range-95-5': 0.0,
        'RRIBasicStat_Range-75-25': 0.0
    }

    if len(qrs_info) >= 3:
        feature['RRIBasicStat_Mean'] = np.mean(qrs_info)
        if feature['RRIBasicStat_Mean'] == 0:
            feature['RRIBasicStat_Hr'] = 0
        else:
            feature['RRIBasicStat_Hr'] = 1 / feature['RRIBasicStat_Mean']

        feature['RRIBasicStat_NumPeaks'] = len(qrs_info)
        feature['RRIBasicStat_Range'] = max(qrs_info) - min(qrs_info)
        feature['RRIBasicStat_Var'] = np.var(qrs_info)
        feature['RRIBasicStat_Std'] = np.std(qrs_info)
        feature['RRIBasicStat_Skewness'] = stats.skew(qrs_info)
        feature['RRIBasicStat_Kurtosis'] = stats.kurtosis(qrs_info)
        feature['RRIBasicStat_Median'] = np.median(qrs_info)
        feature['RRIBasicStat_Min'] = min(qrs_info)
        feature['RRIBasicStat_p5'] = np.percentile(qrs_info, 5)
        feature['RRIBasicStat_p25'] = np.percentile(qrs_info, 25)
        feature['RRIBasicStat_p75'] = np.percentile(qrs_info, 75)
        feature['RRIBasicStat_p95'] = np.percentile(qrs_info, 95)
        feature['RRIBasicStat_Range-95-5'] = feature['RRIBasicStat_p95'] - feature['RRIBasicStat_p5']
        feature['RRIBasicStat_Range-75-25'] = feature['RRIBasicStat_p75'] - feature['RRIBasicStat_p25']

    return feature


def rri_basic_stat_point_median(qrs_info):
    """
    Extract basic statistical features of rr-interval using sliding window
    Refer to: https://github.com/hsd1503/ENCASE
    """
    feature = {
        'RRIBasicStatPointMedian_Mean': 0.0,
        'RRIBasicStatPointMedian_Hr': 0.0,
        'RRIBasicStatPointMedian_Count': 0.0,
        'RRIBasicStatPointMedian_Range': 0.0,
        'RRIBasicStatPointMedian_Var': 0.0,
        'RRIBasicStatPointMedian_Skewness': 0.0,
        'RRIBasicStatPointMedian_Kurtosis': 0.0,
        'RRIBasicStatPointMedian_Median': 0.0,
        'RRIBasicStatPointMedian_Max': 0.0,
        'RRIBasicStatPointMedian_Min': 0.0,
        'RRIBasicStatPointMedian_p25': 0.0,
        'RRIBasicStatPointMedian_p75': 0.0
    }

    # 8-beat sliding window RR interval irregularity detector
    new_qrs_info = []
    for i in range(len(qrs_info) - 2):
        new_qrs_info.append(np.median([qrs_info[i], qrs_info[i + 1], qrs_info[i + 2]]))

    feature['RRIBasicStatPointMedian_Mean'] = np.mean(new_qrs_info)
    if feature['RRIBasicStatPointMedian_Mean'] == 0:
        feature['RRIBasicStatPointMedian_Hr'] = 0
    else:
        feature['RRIBasicStatPointMedian_Hr'] = 1 / feature['RRIBasicStatPointMedian_Mean']

    feature['RRIBasicStatPointMedian_Count'] = len(new_qrs_info)

    if feature['RRIBasicStatPointMedian_Count'] != 0:
        feature['RRIBasicStatPointMedian_Range'] = max(new_qrs_info) - min(new_qrs_info)
        feature['RRIBasicStatPointMedian_Var'] = np.var(new_qrs_info)
        feature['RRIBasicStatPointMedian_Skewness'] = stats.skew(new_qrs_info)
        feature['RRIBasicStatPointMedian_Kurtosis'] = stats.kurtosis(new_qrs_info)
        feature['RRIBasicStatPointMedian_Median'] = np.median(new_qrs_info)
        feature['RRIBasicStatPointMedian_Max'] = max(new_qrs_info)
        feature['RRIBasicStatPointMedian_Min'] = min(new_qrs_info)
        feature['RRIBasicStatPointMedian_p25'] = np.percentile(new_qrs_info, 25)
        feature['RRIBasicStatPointMedian_p75'] = np.percentile(new_qrs_info, 75)
    else:
        pass

    return feature


def rri_basic_stat_deltaRR(qrs_info):
    """
    Extract basic statistical features of the difference between rr-intervals
    Refer to: https://github.com/hsd1503/ENCASE
    """
    feature = {
        'RRIBasicStatDeltaRR_Mean': 0.0,
        'RRIBasicStatDeltaRR_Hr': 0.0,
        'RRIBasicStatDeltaRR_Count': 0.0,
        'RRIBasicStatDeltaRR_Range': 0.0,
        'RRIBasicStatDeltaRR_Var': 0.0,
        'RRIBasicStatDeltaRR_Skewness': 0.0,
        'RRIBasicStatDeltaRR_Kurtosis': 0.0,
        'RRIBasicStatDeltaRR_Median': 0.0,
        'RRIBasicStatDeltaRR_Min': 0.0,
        'RRIBasicStatDeltaRR_p25': 0.0,
        'RRIBasicStatDeltaRR_p75': 0.0
    }

    if len(qrs_info) >= 4:
        qrs_info = np.diff(qrs_info)

        feature['RRIBasicStatDeltaRR_Mean'] = np.mean(qrs_info)

        if feature['RRIBasicStatDeltaRR_Mean'] == 0:
            feature['RRIBasicStatDeltaRR_Hr'] = 0
        else:
            feature['RRIBasicStatDeltaRR_Hr'] = 1 / feature['RRIBasicStatDeltaRR_Mean']

        feature['RRIBasicStatDeltaRR_Count'] = len(qrs_info)
        feature['RRIBasicStatDeltaRR_Range'] = max(qrs_info) - min(qrs_info)
        feature['RRIBasicStatDeltaRR_Var'] = np.var(qrs_info)
        feature['RRIBasicStatDeltaRR_Skewness'] = stats.skew(qrs_info)
        feature['RRIBasicStatDeltaRR_Kurtosis'] = stats.kurtosis(qrs_info)
        feature['RRIBasicStatDeltaRR_Median'] = np.median(qrs_info)
        feature['RRIBasicStatDeltaRR_Min'] = min(qrs_info)
        feature['RRIBasicStatDeltaRR_p25'] = np.percentile(qrs_info, 25)
        feature['RRIBasicStatDeltaRR_p75'] = np.percentile(qrs_info, 75)

    return feature


def cdf(qrs_info):
    """
    Analysis of cumulative distribution functions
    Refer to: https://github.com/hsd1503/ENCASE
    """
    feature = dict()

    n_bins = 60
    hist, _ = np.histogram(qrs_info, range=(100, 400), bins=n_bins)
    cdf = np.cumsum(hist) / len(qrs_info)
    feature['CDF_Density'] = np.sum(cdf) / n_bins

    return feature


def median_absolute_deviation(qrs_info):
    """
    Thresholding on the median absolute deviation (MAD) of RR intervals
    Refer to: https://github.com/hsd1503/ENCASE
    """
    feature = dict()

    peaks_median = np.median(qrs_info)
    feature['Median_Absolute_Deviation'] = np.median([np.abs(peak - peaks_median) for peak in qrs_info])

    return feature


def coeff_of_variation(qrs_info):
    """
    Analysis of cumulative distribution functions
    """
    feature = {
        'CoeffOfVariation_CoeffPeaks': 0.0,
        'CoeffOfVariation_CoeffDiffPeaks': 0.0
    }

    if len(qrs_info) >= 3:
        if np.mean(qrs_info) != 0:
            feature['CoeffOfVariation_CoeffPeaks'] = np.std(qrs_info) / np.mean(qrs_info)

    if len(qrs_info) >= 4:
        diff_qrs_info = np.diff(qrs_info)
        if np.mean(diff_qrs_info) != 0:
            feature['CoeffOfVariation_CoeffDiffPeaks'] = np.std(diff_qrs_info) / np.mean(diff_qrs_info)

    return feature


def hrv_analysis(qrs_info):
    """
    HRV analysis
    """
    feature = {
        'HRV_RMSSD': 0,
        'HRV_NN50': 0,
        'HRV_PNN50': 0,
        'HRV_NN20': 0,
        'HRV_PNN20': 0,
        'HRV_SDNN': 0,
        'HRV_MeanHeartRate': 0,
    }

    if len(qrs_info) > 0:
        diff_qrs_info = np.diff(qrs_info)
        if len(diff_qrs_info) > 0:
            # Root mean square of successive differences
            feature['HRV_RMSSD'] = np.sqrt(np.mean(diff_qrs_info ** 2))
            # Number of pairs of successive NNs that differ by more than 50ms
            feature['HRV_NN50'] = sum(abs(diff_qrs_info) > 50)
            # Proportion of NN50 divided by total number of NNs
            feature['HRV_PNN50'] = (feature['HRV_NN50'] / len(diff_qrs_info)) * 100
            # Number of pairs of successive NNs that differe by more than 20ms
            feature['HRV_NN20'] = sum(abs(diff_qrs_info) > 20)
            # Proportion of NN20 divided by total number of NNs
            feature['HRV_PNN20'] = (feature['HRV_NN20'] / len(diff_qrs_info)) * 100

        # Standard deviation of NN intervals
        feature['HRV_SDNN'] = np.std(qrs_info, ddof=1)  # make it calculates N-1
        # Mean heart rate, in ms
        feature['HRV_MeanHeartRate'] = 60 * 1000.0 / np.mean(qrs_info)

    for key in feature:
        feature[key] = np.round(np.nan_to_num(feature[key]), 4)

    return feature


# test
if __name__ == '__main__':
    b_t = time.time()
    qrs_infos, _, _ = read_qrs_data(folder=os.path.join('..', paths.parsed_train_folder))
    feature_names, long_features = get_qrs_feature(qrs_infos)
    e_t = time.time()

    print("Time to extract qrs features from {0} qrs infos: {1}".format(
        len(qrs_infos), (e_t - b_t)))
    print("Num of features: ", len(long_features[0]))