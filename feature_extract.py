import os
import dill

from read_data import *
from long_features import get_long_feature
from qrs_features import get_qrs_feature
from resnet_features import get_resnet_feature
from cnn_lstm_features import get_cnn_lstm_feature

import paths
from utils import combine_features


def get_features(long_signals=None, qrs_infos=None, expanded_data=None, expanded_name=None,
                 extracted_short_data=None, extracted_short_name=None, name_list=None,
                 model_name=None, cnn_architecture="cnn_feed_lstm", logger=None, **feature_type):

    features = list()

    extract_long_features = feature_type['long']
    extract_qrs_features = feature_type['qrs']
    extract_resnet_features = feature_type['resnet']
    extract_cnn_lstm_features = feature_type['cnn_lstm']


    # calculate features
    if extract_long_features:
        print("Calculating the long features...")
        _, long_features = get_long_feature(long_signals)
        features.append(long_features)
        if logger is not None:
            logger.info("Num of long features: " + str(np.array(long_features).shape[1]))
    

    if extract_qrs_features:
        print("Calculating the qrs features...")
        _, qrs_features = get_qrs_feature(qrs_infos)
        features.append(qrs_features)
        if logger is not None:
            logger.info("Num of qrs features: " + str(np.array(qrs_features).shape[1]))


    if extract_resnet_features:
        print("Calculating the resnet features...")
        resnet_features = get_resnet_feature(
            data=expanded_data, names=expanded_name, name_list=name_list, model_name=model_name["ResNet"]
        )
        features.append(resnet_features)
        if logger is not None:
            logger.info("Num of resnet features: " + str(np.array(resnet_features).shape[1]))

    if extract_cnn_lstm_features:
        print("Calculating the cnn-lstm features...")
        cnn_lstm_features = get_cnn_lstm_feature(
            data=extracted_short_data, names=extracted_short_name, name_list=name_list,
            model_architecture=cnn_architecture, model_name=model_name["CNN-LSTM"]
        )
        features.append(cnn_lstm_features)
        if logger is not None:
            logger.info("Num of cnn-lstm features: " + str(np.array(cnn_lstm_features).shape[1]))

    # combine all calculated features
    all_features = None
    for i in range(len(features)):
        if i == 0:
            all_features = features[i]
        else:
            all_features = combine_features(all_features, features[i])

    if logger is not None:
        logger.info("===========================================================")

    if all_features is not None:
        return all_features
    else:
        raise ValueError("Please choose the valid feature type.")


def save_feature_file(folder=None, filename=None, model_name=None, cnn_architecture="cnn_concat_lstm",
                      data_balance=True, **feature_type):
    """
    Read all data, extract all features and write to dill
    """
    long_signals, long_labels, long_names = None, None, None
    qrs_infos, qrs_labels, qrs_names = None, None, None
    expanded_signals, expanded_labels, expanded_names = None, None, None
    extracted_short_signals, extracted_short_labels, extracted_short_names =  None, None, None

    extract_long_features = feature_type['long']
    extract_qrs_features = feature_type['qrs']
    extract_resnet_features = feature_type['resnet']
    extract_cnn_lstm_features = feature_type['cnn_lstm']

    # load parsed data
    if extract_long_features:
        long_signals, long_labels, long_names = read_balanced_data(folder=folder, datatype='long') if data_balance else read_long_data(folder=folder)

    if extract_qrs_features:
        qrs_infos, qrs_labels, qrs_names = read_balanced_data(folder=folder, datatype='qrs') if data_balance else read_qrs_data(folder=folder)

    # Notice: expanded and extracted data must be obtained from the above loaded data
    if extract_resnet_features:
        if not extract_long_features:
            long_signals, long_labels, long_names = read_balanced_data(folder=folder, datatype='long') if data_balance else read_long_data(folder=folder)
        expanded_signals, expanded_labels, expanded_names = read_expanded_data(folder=folder)

    if extract_cnn_lstm_features:
        if not extract_long_features and not extract_resnet_features:
            long_signals, long_labels, long_names = read_balanced_data(folder=folder, datatype='long') if data_balance else read_long_data(folder=folder)
        extracted_short_signals, extracted_short_labels, extracted_short_names, _ = read_extracted_short_data(folder=folder)

    # get the name list -- be used for combine the calculated deep features
    name_list = dict()
    for i in range(len(long_names)):
        current_name = long_names[i]
        name_list[current_name] = i

    # get all features
    all_features = get_features(long_signals=long_signals, qrs_infos=qrs_infos,
                                expanded_data=expanded_signals, expanded_name=expanded_names,
                                extracted_short_data=extracted_short_signals, extracted_short_name=extracted_short_names,
                                name_list=name_list, model_name=model_name, cnn_architecture=cnn_architecture,
                                logger=None, **feature_type)

    print("Feature extraction is completed.")
    print("All features shape:", np.array(all_features).shape)

    all_names, all_labels = qrs_names, qrs_labels
    with open(folder + filename, 'wb') as file_feature:
        dill.dump(all_names, file_feature)
        dill.dump(all_features, file_feature)
        dill.dump(all_labels, file_feature)
    print("Write completed.")

    return


if __name__ == '__main__':

    parsed_train_folder = paths.parsed_train_folder
    feature_type = dict()

    # ------------------------------------ NEED TO DEFINE --------------------------------- #
    base_name = "balance_long129_filterd"  # base name of feature file and classifier to save

    # choose whether use the balanced data
    data_balance = True

    # choose the type of features to be calculated
    feature_type['long'] = True
    feature_type['qrs'] = False
    feature_type['resnet'] = False
    feature_type['cnn_lstm'] = False

    # specify the pre-trained model
    resnet_name = "resnet_balance_06-11-10_57_52.pkl"
    cnn_lstm_name = "cnn_concat_lstm_0608_031124.pkl"
    # ------------------------------------ NEED TO DEFINE --------------------------------- #

    model_name = dict()
    model_name["ResNet"] = os.path.join(paths.models_path, resnet_name)
    model_name["CNN-LSTM"] = os.path.join(paths.models_path, cnn_lstm_name)
    cnn_architecture = "cnn_concat_lstm" if "cnn_concat_lstm" in cnn_lstm_name else "cnn_feed_lstm"

    feature_filename = "features_" + base_name + ".pkl"
    save_feature_file(folder=parsed_train_folder, filename=feature_filename, model_name=model_name,
                      cnn_architecture=cnn_architecture, data_balance=True, **feature_type)

