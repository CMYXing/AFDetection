import os
import random

import csv
import numpy as np
from collections import Counter

import fastdtw
from sklearn import cluster
from ecgdetectors import Detectors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import paths
import utils
from wettbewerb import load_references
from read_data import read_long_data, read_short_data, read_balanced_data
import augmentation as aug


def get_long_qrs_short_data(folder, output_path):
    """
    Load the raw data, write the long/qrs/short data in .csv file
    """

    file_short = open(os.path.join(output_path, "short.csv"), 'w')
    file_long = open(os.path.join(output_path, "long.csv"), 'w')
    file_qrs = open(os.path.join(output_path, "qrs_info.csv"), 'w')
    writer_short = csv.writer(file_short, delimiter=',')
    writer_long = csv.writer(file_long, delimiter=',')
    writer_qrs = csv.writer(file_qrs, delimiter=',')

    ecg_leads, ecg_labels, fs, ecg_names = load_references(folder=folder)
    num_samples = len(ecg_leads)

    detector = Detectors(sampling_frequency=fs)  # initialize a QRS detector
    n_iter = 10  # Maximum number of QRS detections per signal

    for i in range(num_samples):
        name, label, signal = ecg_names[i], ecg_labels[i], ecg_leads[i]

        r_peaks = detector.swt_detector(signal)  # calculate the QRS data
        r_peaks.insert(0, 0)
        r_peaks.append(ecg_leads[i].shape[0])
        qrs_info = list(np.diff(r_peaks))

        ### Recursive QRS detection
        # The swt detector sometimes output long sequence composed of several undivided QRS.
        # To handle this, we apply the swt detector on the unsegmented long sequence one more time.
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

        # write long data
        list_long = []
        list_long.append(name)
        list_long.append(label)
        list_long.extend(signal)
        writer_long.writerow(list_long)

        # write short data
        # Notice: Some signals cannot accurately determine the peaks due to noise, so have no short data
        for j in range(2, len(r_peaks) - 2):
            list_short = []
            list_short.append(name)
            list_short.append(label)
            segment = signal[r_peaks[j] + 1:r_peaks[j + 1]]  # +1 to avoid overlap
            list_short.extend(segment)
            writer_short.writerow(list_short)

        # write QRS data
        list_qrs = []
        list_qrs.append(name)
        list_qrs.append(label)
        # add 0 and length to head and tail, diff to get length of each segment,
        # notice that the first and the last is not accurate
        list_qrs.extend(np.array(qrs_info))
        writer_qrs.writerow(list_qrs)

    file_long.close()
    file_short.close()
    file_qrs.close()
    print("Write long data is done.")
    print("Write short data is done.")
    print("Write qrs data is done.\n")

    return


def get_balanced_data(folder, class_to_be_balanced):

    # Notice: Only the training data can be balanced
    file_long = open(os.path.join(folder, "balanced_long.csv"), 'w')
    file_short = open(os.path.join(folder, "balanced_short.csv"), 'w')
    file_qrs = open(os.path.join(folder, "balanced_qrs.csv"), 'w')
    writer_long = csv.writer(file_long, delimiter=',')
    writer_short = csv.writer(file_short, delimiter=',')
    writer_qrs = csv.writer(file_qrs, delimiter=',')

    long_signals, long_labels, long_names = read_long_data(folder=folder)
    fs = 300

    balanced_signals, balanced_labels, balanced_names = long_signals, long_labels, long_names
    for class_tuple in class_to_be_balanced:
        balanced_signals, balanced_labels, balanced_names = aug.balance_minority_class(
            balanced_signals, balanced_labels, balanced_names, class_tuple
        )

    # shuffle the balanced long data
    balanced_data = list(zip(balanced_signals, balanced_labels, balanced_names))
    random.shuffle(balanced_data)
    balanced_signals[:], balanced_labels[:], balanced_names[:] = zip(*balanced_data)

    # recalculated the short/qrs data
    num_samples = len(balanced_names)
    detector = Detectors(sampling_frequency=fs)  # initialize a QRS detector
    n_iter = 10  # Maximum number of QRS detections per signal

    for i in range(num_samples):
        name, label, signal = balanced_names[i], balanced_labels[i], np.array(balanced_signals[i])

        r_peaks = detector.swt_detector(signal)  # calculate the QRS data
        r_peaks.insert(0, 0)
        r_peaks.append(len(balanced_signals[i]))
        qrs_info = list(np.diff(r_peaks))

        ### Recursive QRS detection
        # The swt detector sometimes output long sequence composed of several undivided QRS.
        # To handle this, we apply the swt detector on the unsegmented long sequence one more time.
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

        # write long data
        list_long = []
        list_long.append(name)
        list_long.append(label)
        list_long.extend(signal)
        writer_long.writerow(list_long)

        # write short data
        # Notice: Some signals cannot accurately determine the peaks due to noise, so have no short data
        for j in range(2, len(r_peaks) - 2):
            list_short = []
            list_short.append(name)
            list_short.append(label)
            segment = signal[r_peaks[j] + 1:r_peaks[j + 1]]  # +1 to avoid overlap
            list_short.extend(segment)
            writer_short.writerow(list_short)

        # write QRS data
        list_qrs = []
        list_qrs.append(name)
        list_qrs.append(label)
        # add 0 and length to head and tail, diff to get length of each segment,
        # notice that the first and the last is not accurate
        list_qrs.extend(np.array(qrs_info))
        writer_qrs.writerow(list_qrs)

    file_long.close()
    file_short.close()
    file_qrs.close()
    print("Write balanced long data is done.")
    print("Write balanced short data is done.")
    print("Write balanced qrs data is done.\n")


def get_expanded_data(folder, train=True, data_balance=True, mode="entire"):
    """
    Load the long data, write the expanded data in .csv file

    :param mode: str, optional: ["two_part", "entire"]
        "entire"    -> expand the data directly
        "two_part"  -> split the data into 2 parts (train, test), and expand them respectively
    """
    class_balance = False
    if data_balance:
        assert train  # make sure that only the training data can be balanced
        long_signals, long_labels, long_names = read_balanced_data(folder=folder, datatype='long')
    else:
        long_signals, long_labels, long_names = read_long_data(folder=folder)
        if train:
            class_balance = True

    if mode == "two_part":
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(np.array(long_signals), np.array(long_labels)):
            train_signal = np.array(long_signals)[train_index]
            train_label = np.array(long_labels)[train_index]
            train_name = np.array(long_names)[train_index]
            test_signal = np.array(long_signals)[test_index]
            test_label = np.array(long_labels)[test_index]
            test_name = np.array(long_names)[test_index]
            break

        train_signal, train_label, train_name = utils.get_segments(
            list(train_signal), list(train_label), list(train_name), class_balance=class_balance
        )
        test_signal, test_label, test_name = utils.get_segments(
            list(test_signal), list(test_label), list(test_name), class_balance=class_balance
        )

        with open(folder + 'expanded_data_2part.bin', 'wb') as file:
            np.save(file, train_signal)
            np.save(file, train_label)
            np.save(file, train_name)
            np.save(file, test_signal)
            np.save(file, test_label)
            np.save(file, test_name)
            file.close()

    else:
        segmented_signals, segmented_labels, segmented_names = utils.get_segments(
            long_signals, long_labels, long_names, class_balance=class_balance
        )

        with open(folder + 'expanded_data.bin', 'wb') as file:
            np.save(file, segmented_signals)
            np.save(file, segmented_labels)
            np.save(file, segmented_names)
            file.close()

    print("Write expanded data is done.\n")


def get_extracted_short_data(folder, train=True, data_balance=True, cluster=False, mode="entire"):

    # read the long/short data
    if data_balance:
        _, long_labels, long_names = read_balanced_data(folder=folder, datatype='long')
        short_signals, _, short_names = read_balanced_data(folder=folder, datatype='short')
    else:
        _, long_labels, long_names = read_long_data(folder=folder)
        short_signals, _, short_names = read_short_data(folder=folder)

    # set parameters for selecting
    resampled_length = 1000
    n_clusters = 4
    radius = 1

    extracted_short_signals = list()
    extracted_short_labels = list()
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
        step += 1

        # select name who has more than 5 short signals
        if name in short_names_dict.keys() and len(short_names_dict[name]) > 5:

            # sub_short_signals: contains all short signals of the current name
            sub_short_signals = short_names_dict[name]

            # choose if filter the raw short signals via cluster
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

        idx = long_names.index(name)
        label = long_labels[idx]

        # combine each four selected short signals
        spliced_siganls = list()
        if len(selected_signals) < 4:
            spliced_siganls = [[0.0] * 1000]
        else:
            if train and not data_balance:
                # for training data (class balancing)
                if label == 'N':
                    for i in range(0, len(selected_signals), 4):
                        spliced_signal = list()
                        if (i + 3) > (len(selected_signals) - 1):
                            break
                        for j in range(4):
                            spliced_signal.extend(selected_signals[i + j])
                        spliced_siganls.append(spliced_signal)

                elif label == 'O':
                    for i in range(0, len(selected_signals), 2):
                        spliced_signal = list()
                        if (i + 3) > (len(selected_signals) - 1):
                            break
                        for j in range(4):
                            spliced_signal.extend(selected_signals[i + j])
                        spliced_siganls.append(spliced_signal)

                else:  # 'A' or '~'
                    for i in range(0, len(selected_signals)):
                        spliced_signal = list()
                        if (i + 3) > (len(selected_signals) - 1):
                            break
                        for j in range(4):
                            spliced_signal.extend(selected_signals[i + j])
                        spliced_siganls.append(spliced_signal)
            else:
                for i in range(0, len(selected_signals)):
                    spliced_signal = list()
                    if (i + 3) > (len(selected_signals) - 1):
                        break
                    for j in range(4):
                        spliced_signal.extend(selected_signals[i + j])
                    spliced_siganls.append(spliced_signal)

        # resample the spliced signals
        resampled_signals = utils.resample_signals(spliced_siganls, length=resampled_length)

        if step % 1000 == 0:
            print('Step: ', step, '/', len(long_names))

        extracted_short_signals.extend(resampled_signals)
        extracted_short_names.extend([name] * len(resampled_signals))
        extracted_short_labels.extend([label] * len(resampled_signals))

    index = np.array(list(range(len(extracted_short_names))))
    extracted_short_signals = np.expand_dims(np.array(extracted_short_signals, dtype=np.float32), axis=1)
    extracted_short_names = np.array(extracted_short_names, dtype=np.string_)

    le = LabelEncoder()
    extracted_short_labels = le.fit_transform(np.array(extracted_short_labels))

    # Shuffle
    index_shuffle = np.random.permutation(index)
    extracted_short_signals = extracted_short_signals[index_shuffle]
    extracted_short_labels = extracted_short_labels[index_shuffle]
    extracted_short_names = extracted_short_names[index_shuffle]

    if mode == "two_part":
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(np.array(extracted_short_signals), np.array(extracted_short_labels)):
            train_signal = np.array(extracted_short_signals)[train_index]
            train_label = np.array(extracted_short_labels)[train_index]
            train_name = np.array(extracted_short_names)[train_index]
            test_signal = np.array(extracted_short_signals)[test_index]
            test_label = np.array(extracted_short_labels)[test_index]
            test_name = np.array(extracted_short_names)[test_index]
            break

        with open(folder + 'extracted_data_2part.bin', 'wb') as file:
            np.save(file, train_signal)
            np.save(file, train_label)
            np.save(file, train_name)
            np.save(file, test_signal)
            np.save(file, test_label)
            np.save(file, test_name)
            file.close()

    else:
        with open(folder + 'extracted_short.bin', 'wb') as file:
            np.save(file, extracted_short_signals)
            np.save(file, extracted_short_labels)
            np.save(file, extracted_short_names)
            file.close()

    print("Write extracted short data is done.\n")
    return


if __name__ == "__main__":

    # set load and save path
    train_folder = paths.original_train_folder
    train_output_path = paths.parsed_train_folder

    ### Prepare the training data
    # generate the long/qrs/short data
    get_long_qrs_short_data(folder=train_folder, output_path=train_output_path)

    data_balance = False
    # generate the expanded data
    class_to_be_balanced = [('N', 'A'), ('N', 'O'), ('N', '~')]
    if class_to_be_balanced is not None:
        data_balance = True
        # generate the balanced data
        get_balanced_data(folder=train_output_path, class_to_be_balanced=class_to_be_balanced)
        get_expanded_data(folder=train_output_path, train=True, data_balance=data_balance)
    else:
        get_expanded_data(folder=train_output_path, train=True, data_balance=data_balance)

    # generate the extracted short data
    get_extracted_short_data(folder=train_output_path, train=True, data_balance=True)

