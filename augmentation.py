import numpy as np
from collections import Counter
from imblearn.over_sampling import ADASYN


### Offline data augmentation
def balance_minority_class(long_signals, long_labels, long_names, class_tuple):
    """
    class_to_be_balanced: tuple, contains a majority (first) and a minority class (second)
                          e.g., ('N', 'A')
    """
    minority_class = class_tuple[1]

    balanced_signals = list()
    balanced_labels = list()
    balanced_names = list()

    length = 9000  # here choose the original signals with a certain length (majority of signals are 9000 in length)

    for i in range(len(long_labels)):  # this part of the data will not be augmented
        if len(long_signals[i]) != length or long_labels[i] != minority_class:
            balanced_signals.append(long_signals[i])
            balanced_labels.append(long_labels[i])
            balanced_names.append(long_names[i])

    selected_signals, selected_labels = select_same_length_data(
        long_signals, long_labels, length=length, specified_label=class_tuple
    )
    resampled_signals, resampled_labels = adaptive_synthetic_sampling(
        selected_signals, selected_labels, show=True
    )
    minority_class_signals, minority_class_labels = get_minority_class_data(
        resampled_signals, resampled_labels, minority_class
    )

    balanced_signals.extend(minority_class_signals)
    balanced_labels.extend(minority_class_labels)

    for i in range(len(minority_class_labels)):
        balanced_names.append("balanced_" + minority_class + "_" + str(i))

    return balanced_signals, balanced_labels, balanced_names


def select_same_length_data(long_signals, long_labels, length=9000, specified_label=None):
    """
    specified_label: list of specified labels, e.g., ('N', 'A')
    """
    signals = list()
    labels = list()

    try:
        for i in range(len(long_signals)):
            if long_labels[i] == specified_label[0] and len(long_signals[i]) == length:
                signals.append(long_signals[i])
                labels.append(long_labels[i])
            elif long_labels[i] == specified_label[1] and len(long_signals[i]) == length:
                signals.append(long_signals[i])
                labels.append(long_labels[i])
    except:
        raise ValueError("Unsupported label.")

    if len(list(set(labels))) < 2:
        raise ValueError("Invalid data length.")

    return signals, labels


def adaptive_synthetic_sampling(signals, labels, show=False):

    ada = ADASYN(random_state=42, n_neighbors=50)
    balanced_signals, balanced_labels = ada.fit_resample(signals, labels)

    if show:
        print('Original dataset: %s' % Counter(labels))
        print('Balanced dataset: %s' % Counter(balanced_labels))

    return balanced_signals,  balanced_labels


def get_minority_class_data(balanced_signals, balanced_labels, minority_class):

    minority_class_signals = list()
    minority_class_labels = list()

    for i in range(len(balanced_labels)):
        if balanced_labels[i] == minority_class:
            minority_class_signals.append(balanced_signals[i])
            minority_class_labels.append(balanced_labels[i])

    return minority_class_signals, minority_class_labels


###  Online data augmentation
def data_augmentation(aug_params):
    def augmentation(signal):
        sigma = aug_params['sigma']
        interval = aug_params['shift_interval']

        # if np.random.randn() > 0.5:
        #     signal = verflip(signal)
        if np.random.randn() > 0.5:
            signal = scaling(signal, sigma=sigma)
        if np.random.randn() > 0.5:
            signal = shift(signal, interval=interval)
        return signal

    return augmentation


def verflip(signal):
    return signal[:, ::-1].copy()


def scaling(signal, sigma=0.1):
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(1, signal.shape[0]))
    return signal * scaling_factor


def shift(signal, interval=20):
    for row in range(signal.shape[0]):
        offset = np.random.choice(range(-interval, interval))
        signal[row, :] += offset
    return signal








