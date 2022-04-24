import os
import time
import logging
import warnings
warnings.filterwarnings("ignore")

import dill
import pickle

import numpy as np
from sklearn.model_selection import StratifiedKFold

import paths
from utils import evaluation
from classifier import XGBoost


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def training(filename, model_name=None):
    try:
        with open(os.path.join(paths.parsed_train_folder, filename), 'rb') as input:
            all_names = dill.load(input)
            all_features = dill.load(input)
            all_labels = dill.load(input)

        # Logging
        handler = logging.FileHandler("logging/xgboost_" + time.strftime("%Y-%m-%d-%H:%M:%S") + ".txt")
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.info("===========================================================")
        logger.info("Training data: " + os.path.join(paths.parsed_train_folder, filename))

        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        print("All features shape: {0}".format(all_features.shape))

        logger.info("All features shape: {0}".format(all_features.shape))
        logger.info("Model name: " + model_name)

        # k-fold cross validation
        f1_list = []
        k_folds = StratifiedKFold(n_splits=5, shuffle=True)
        i_fold = 1
        for train_index, test_index in k_folds.split(all_features, all_labels):
            train_data = all_features[train_index]
            train_label = all_labels[train_index]
            test_data = all_features[test_index]
            test_label = all_labels[test_index]

            # train the classifier
            classifier = XGBoost.load_classifier()
            classifier.fit(train_data, train_label)

            # predict the training data and evaluate the results
            train_pred = classifier.predict(train_data)
            train_f1 = evaluation(train_label, train_pred)

            # predict the test data and evaluate the results
            test_pred = classifier.predict(test_data)
            test_f1, report = evaluation(test_label, test_pred, report=True)

            print(i_fold, '-fold: ', "Train f1 score: ", round(train_f1, 2),
                  "; Test f1 score:", round(test_f1, 2))
            print(report)
            f1_list.append(test_f1)

            logger.info("{0}-fold: Train f1 score {1}; Test f1 score {2}".format(
                i_fold, round(train_f1, 4), round(test_f1, 4)))
            logger.info(report)
            logger.info("-----------------------------------------------------------")

            i_fold += 1

        # train on the entire data and save
        classifier = XGBoost.load_classifier()
        classifier.fit(all_features, all_labels)

        pickle.dump(classifier.boost, open(os.path.join(paths.models_path, model_name), "wb"))
        print("Classification model has been saved.")
        logger.info("The trained XGBoost classifier has been saved.")

    except:
        raise FileExistsError("The current filename do not exist.")


def classification(features, names, model_name=None):

    # load the xgboost classifier
    classifier = XGBoost.load_classifier(path=model_name)

    # make predications
    results = classifier.predict(features)

    label_dict = {0: 'A', 1: 'N', 2: 'O', 3: '~'}
    predictions = list()
    pred = list()  # for calculating f1-score
    for i in range(len(results)):
        pred.append(label_dict[results[i]])
        predictions.append((names[i], label_dict[results[i]]))

    return pred, predictions


if __name__ == '__main__':

    feature_file = "features_balance_long113.pkl"
    model_name = "xgboost_balance_long113.dat"
    training(filename=feature_file, model_name=model_name)

