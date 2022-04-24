# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import os
import time
import logging

import numpy as np
import pandas as pd

from parse_data import get_parsed_test_data
from feature_extract import get_features
from classification import classification

import utils
import paths


### Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads, fs, ecg_names, use_pretrained=False):
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.

    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''
    assert use_pretrained

    record = False
    calc_f1 = False

    # choose the feature type to calculate
    feature_type = dict()
    feature_type['long'] = True
    feature_type['qrs'] = True
    feature_type['resnet'] = True
    feature_type['cnn_lstm'] = True

    # Pre-trained model setting
    model_name = None
    if use_pretrained:
        model_name = dict()
        model_name["Classifier"] = os.path.join(paths.models_path, "xgboost_balance_long_qrs_res_cnn_feed1.dat")
        model_name["ResNet"] = os.path.join(paths.models_path, "resnet_06-14-11_41_26.pkl")
        model_name["CNN-LSTM"] = os.path.join(paths.models_path, "cnn_feed_lstm_06-14-00_21_06.pkl")
        cnn_architecture = "cnn_concat_lstm" if "cnn_concat_lstm" in model_name["CNN-LSTM"] else "cnn_feed_lstm"

    logger = None
    if record:
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler("logging/final_test_" + time.strftime("%m-%d-%H:%M:%S") + ".txt")
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        logger.info("========================= Feature =========================")
        logger.info("Long features: " + str(feature_type['long']))
        logger.info("QRS features: " + str(feature_type['qrs']))
        logger.info("ResNet features: " + str(feature_type['resnet']))
        logger.info("CNN-LSTM features: " + str(feature_type['cnn_lstm']))

        logger.info("========================= Setting =========================")
        logger.info("Num of test samples: " + str(len(ecg_names)))
        logger.info("Classifier model: " + str(model_name["Classifier"]))
        if feature_type['resnet']:
            logger.info("ResNet model: " + str(model_name["ResNet"]))
        if feature_type['cnn_lstm']:
            logger.info("CNN-LSTM model: " + str(model_name["CNN-LSTM"]))
            logger.info("CNN-LSTM model architecture: " + str(cnn_architecture))
        logger.info("======================== Features =========================")

    # Prediction starts here
    pred_b = time.time()

    print("Preparing the data...")
    long_signals, long_names, \
    qrs_infos, qrs_names, \
    _, _, \
    expanded_signals, expanded_names, \
    extracted_short_signals, extracted_short_names = get_parsed_test_data(ecg_leads, ecg_names, fs, logger=logger)

    name_list = {}  # be used for the fusion of deep features
    for i in range(len(ecg_names)):
        current_name = ecg_names[i]
        name_list[current_name] = i

    features = get_features(long_signals=long_signals, qrs_infos=qrs_infos,
                            expanded_data=expanded_signals, expanded_name=expanded_names,
                            extracted_short_data=extracted_short_signals, extracted_short_name=extracted_short_names,
                            name_list=name_list, model_name=model_name, cnn_architecture=cnn_architecture,
                            logger=logger, **feature_type)
    features = np.array(features)

    print("------------------------------------------------------------")
    print("All features (shape: {0}, {1}) have been extracted."
          "\nStarting to classify...".format(features.shape[0], features.shape[1]))

    # Classification
    pred, predictions = classification(features=features, names=ecg_names, model_name=model_name["Classifier"])

    pred_e = time.time()

    if calc_f1:
        # load ground truth
        data = pd.read_csv(paths.test_csv_path, header=None).values
        true = list(data[:, 1])

        # calculate the f1 score
        f1_score, report = utils.evaluation(true, pred, report=True)
        print("F1-score: ", f1_score)
        print(report)

        if logger is not None:
            logger.info(report)

    if logger is not None:
        logger.info("===========================================================")
        logger.info("Prediction time: " + str(pred_e - pred_b))

    #------------------------------------------------------------------------------
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
