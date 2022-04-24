import numpy as np
from collections import Counter

import fastdtw
from sklearn import cluster
from ecgdetectors import Detectors

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report


def combine_features(f1, f2):
    features = []

    # f1 and f2 should have the same length
    n_row = len(f1)
    for i in range(n_row):
        feature_1 = list(f1[i])
        feature_2 = list(f2[i])
        features.append(feature_1 + feature_2)

    return features


def resample_signals(signals, length=1000):
    resampled_signals = list()

    for signal in signals:
        if len(signal) != length:
            x = np.arange(len(signal))
            new_x = np.arange(0, len(signal)-1, (len(signal)-1) / length)
            resampled_signal = list(np.interp(new_x, x, signal))
            resampled_signal = resampled_signal[:1000]
            resampled_signals.append(resampled_signal)
        else:
            resampled_signals.append(signal)

    return resampled_signals


def get_segments(signals=None, labels=None, names=None, class_balance=True):
    """
    Segment the long data
    Refer to: https://github.com/hsd1503/ENCASE (a bit modified)
    """
    seg_signals = list()
    seg_labels = list()
    seg_names = list()

    seg_length = 6000
    initial_stride = 500

    if class_balance:
        counter = {'N': 0, 'O': 0, 'A': 0, '~': 0}
        for i in range(len(labels)):
            counter[labels[i]] += len(signals[i])

        stride_N = initial_stride
        stride_O = int(stride_N // (counter['N'] / counter['O']))
        stride_A = int(stride_N // (counter['N'] / counter['A']))
        stride_P = int(0.85 * stride_N // (counter['N'] / counter['~']))

        stride = {'N': stride_N, 'O': stride_O, 'A': stride_A, '~': stride_P}
    else:
        stride = {'N': initial_stride, 'O': initial_stride, 'A': initial_stride, '~': initial_stride}

    for i in range(len(signals)):
        temp_stride = stride[labels[i]]
        temp_signal = signals[i]
        while len(temp_signal) <= seg_length:
            temp_signal.extend(np.array([0.]))
        for j in range(0, len(temp_signal) - seg_length, temp_stride):
            seg_signals.append(temp_signal[j:j + seg_length])
            if names is not None:
                seg_names.append(names[i])
            if labels is not None:
                seg_labels.append(labels[i])

    index = np.array(list(range(len(seg_signals))))
    seg_signals = np.expand_dims(np.array(seg_signals, dtype=np.float32), axis=1)

    if names is not None:
        seg_names = np.array(seg_names, dtype=np.string_)

    if labels is not None:
        le = LabelEncoder()
        seg_labels = le.fit_transform(np.array(seg_labels))

    # Shuffle
    index_shuffle = np.random.permutation(index)
    seg_signals = seg_signals[index_shuffle]
    if labels is not None:
        seg_labels = seg_labels[index_shuffle]
    if names is not None:
        seg_names = seg_names[index_shuffle]

    return seg_signals, seg_labels, seg_names


def evaluation(true, pred, report=False):
    """
            N     A     O     ~    Total
    A	    Nn    Na    No	  Np    ∑N
    N	    An    Aa    Ao    Ap	∑A
    O	    On	  Oa	Oo    Op	∑O
    ~	    Pn	  Pa	Po	  Pp	∑P
    Total   ∑n    ∑a 	∑o	  ∑p
    """
    f1 = f1_score(true, pred, average='micro')

    if report:
        report = classification_report(true, pred, target_names=['A', 'N', 'O', '~'])
        return f1, report
    else:
        return f1

