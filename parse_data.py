import numpy as np
from collections import Counter

import fastdtw
from sklearn import cluster
from ecgdetectors import Detectors

import utils


def get_parsed_test_data(ecg_leads, ecg_names, fs, logger=None):

    long_signals, long_names = list(), list()
    short_signals, short_names = list(), list()
    qrs_infos, qrs_names = list(), list()

    num_samples = len(ecg_leads)

    detector = Detectors(sampling_frequency=fs)  # initialize a QRS detector
    n_iter = 10  # Maximum number of QRS detections per signal

    print("Get long/qrs/short data...")
    for i in range(num_samples):
        name, signal = ecg_names[i], ecg_leads[i]

        r_peaks = detector.swt_detector(signal)  # calculate the QRS data
        r_peaks.insert(0, 0)
        r_peaks.append(ecg_leads[i].shape[0])
        qrs_info = list(np.diff(r_peaks))

        # Recursive QRS detection
        indices = [qrs_info.index(value) for value in qrs_info if value > (2 * fs)]
        iter = 0
        while len(indices) > 0:
            iter += 1
            if iter >= n_iter:
                break

            index = indices[0]
            part_signal = signal[r_peaks[index]:r_peaks[index + 1]]
            part_r_peaks = detector.swt_detector(part_signal)
            part_r_peaks = [i + r_peaks[index] for i in part_r_peaks]

            r_peaks = r_peaks[:index + 1] + part_r_peaks[:-1] + r_peaks[index + 1:]
            qrs_info = list(np.diff(r_peaks))
            indices = [qrs_info.index(value) for value in qrs_info if value > (2 * fs)]

        # Get long data
        long_signals.append(signal)
        long_names.append(name)

        # Get short data
        for j in range(2, len(r_peaks) - 2):
            short_names.append(name)
            segment = signal[r_peaks[j] + 1:r_peaks[j + 1]]  # +1 to avoid overlap
            short_signals.append(segment)

        # Get QRS data
        qrs_names.append(name)
        qrs_infos.append(np.array(qrs_info))

    expanded_signals, expanded_names, extracted_short_signals, extracted_short_names = None, None, None, None

    # Get expanded data
    signals = [list(signal) for signal in ecg_leads]
    names = list(ecg_names)
    expanded_signals, expanded_names = get_expanded_test_data(signals=signals, names=names, logger=logger)

    # Get extracted short data
    extracted_short_signals, extracted_short_names = get_extracted_test_data(
        short_signals=short_signals, short_names=short_names, long_names=long_names
    )

    return long_signals, long_names, \
           qrs_infos, qrs_names, \
           short_signals, short_names, \
           expanded_signals, expanded_names, \
           extracted_short_signals, extracted_short_names


def get_expanded_test_data(signals=None, names=None, logger=None):
    """
    Expand the test data, use a uniform stride value of 100
    """
    print("Get expanded data from long data...")
    seg_signals = list()
    seg_names = list()

    seg_length = 6000
    stride = 500  # might be reduced, because inference time is too long

    if logger is not None:
        logger.info("Stride of the expanded data: " + str(stride))

    for i in range(len(signals)):
        temp_signal = signals[i]
        while len(temp_signal) <= seg_length:
            temp_signal.extend(np.array([0.]))

        for j in range(0, len(temp_signal) - seg_length, stride):
            seg_signals.append(temp_signal[j:j + seg_length])
            seg_names.append(names[i])

    seg_signals = np.expand_dims(np.array(seg_signals, dtype=np.float32), axis=1)
    seg_names = np.array(seg_names, dtype=np.string_)

    return seg_signals, seg_names


def get_extracted_test_data(short_signals=None, short_names=None, long_names=None, cluster=False):

    print("Get extracted data from long and short data...")

    # set parameters for selecting
    resampled_length = 1000
    n_clusters = 4
    radius = 1

    extracted_short_signals = list()
    extracted_short_names = list()

    short_names_dict = dict()
    # build the dictionary for short names
    for i in range(len(short_names)):
        if short_names[i] in short_names_dict.keys():
            short_names_dict[short_names[i]].append(short_signals[i])
        else:
            short_names_dict[short_names[i]] = [short_signals[i]]

    step = 0
    for name in long_names:

        # select name who has more than 2 short signals
        if name in short_names_dict.keys() and len(short_names_dict[name]) > 5:
            # sub_short_signals contains all short signals of the current name
            sub_short_signals = short_names_dict[name]

            if cluster:
                # construct distance matrix of short signals
                len_short = len(sub_short_signals)
                distance_matrix = np.zeros([len_short, len_short])
                for i in range(len_short):
                    distance_matrix[i, i] = 0.0
                    for j in range(i + 1, len_short):
                        # the approximate distance between the 2 time series: abs(x[i] - y[j])
                        tmp_distance = fastdtw.fastdtw(sub_short_signals[i], sub_short_signals[j], radius=radius)[0]
                        distance_matrix[i, j] = tmp_distance
                        distance_matrix[j, i] = tmp_distance

                # K-means cluster via distance matrix
                clustering = cluster.KMeans(n_clusters=n_clusters, random_state=4).fit(distance_matrix)

                # Find the most common label
                majority_label = Counter(clustering.labels_).most_common(2)[0][0]
                selected_short_index = np.array(list(range(len_short)))[clustering.labels_ == majority_label]

                selected_signals = list()
                for index in selected_short_index:
                    selected_signals.append(sub_short_signals[index])
            else:
                selected_signals = sub_short_signals

        else:
            selected_signals = [[0.0] * 1000]

        # combine each four selected short signals
        spliced_siganls = list()
        if len(selected_signals) < 4:
            spliced_siganls = [[0.0] * 1000]
        else:
            for i in range(0, len(selected_signals), 4):
                spliced_signal = list()
                if (i + 3) > (len(selected_signals) - 1):
                    break
                for j in range(4):
                    spliced_signal.extend(selected_signals[i + j])
                spliced_siganls.append(spliced_signal)

        # resample the splice signals
        resampled_signals = utils.resample_signals(spliced_siganls, length=resampled_length)
        resampled_names = [name] * len(resampled_signals)

        step += 1
        if step % 1000 == 0:
            print('Step: ', step, '/', len(long_names))

        extracted_short_signals.extend(resampled_signals)
        extracted_short_names.extend(resampled_names)

    extracted_short_signals = np.expand_dims(np.array(extracted_short_signals, dtype=np.float32), axis=1)
    extracted_short_names = np.array(extracted_short_names, dtype=np.string_)

    return extracted_short_signals, extracted_short_names