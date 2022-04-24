import os

from prepare_dataset import *
from feature_extract import save_feature_file
import classification
import paths



if __name__ == '__main__':

    ### Generate training dataset
    ## Training data
    # set load and save path
    train_folder = paths.original_train_folder
    parsed_train_folder = paths.parsed_train_folder
    test_folder = paths.original_test_folder
    parsed_test_path = paths.parsed_test_folder

    # generate the long/qrs/short data
    get_long_qrs_short_data(folder=train_folder, output_path=parsed_train_folder)

    data_balance = False
    # generate the expanded data
    class_to_be_balanced = [('N', 'A'), ('N', 'O'), ('N', '~')]
    if class_to_be_balanced is not None:
        data_balance = True
        # generate the balanced data
        get_balanced_data(folder=parsed_train_folder, class_to_be_balanced=class_to_be_balanced)
        get_expanded_data(folder=parsed_train_folder, train=True, data_balance=data_balance, mode="entire")
    else:
        get_expanded_data(folder=parsed_train_folder, train=True, data_balance=data_balance, mode="entire")

    # generate the extracted short data
    get_extracted_short_data(folder=parsed_train_folder, train=True, data_balance=data_balance, mode="entire")


    ## Test data
    # generate the long/qrs/short data
    get_long_qrs_short_data(folder=test_folder, output_path=parsed_test_path)

    # generate the expanded data
    get_expanded_data(folder=parsed_test_path, train=False, data_balance=False, mode="entire")

    # generate the extracted short data
    get_extracted_short_data(folder=parsed_test_path, train=False, data_balance=False, mode="entire")



    # ### Generate the feature file -- be used for training the classifier
    # feature_type = dict()
    #
    # # ------------------------------------ NEED TO DEFINE --------------------------------- #
    # base_name = "balance_long_qrs_res_cnn"  # base name of feature file and classifier to save
    #
    # # choose whether use the balanced data
    # data_balance = True
    #
    # # choose the type of features to be calculated
    # feature_type['long'] = True
    # feature_type['qrs'] = True
    # feature_type['resnet'] = True
    # feature_type['cnn_lstm'] = True
    #
    # # specify the pre-trained model
    # resnet_name = "resnet_06-14-11:41:26.pkl"
    # cnn_lstm_name = "cnn_feed_lstm_06-14-19:43:55.pkl"
    # # ------------------------------------ NEED TO DEFINE --------------------------------- #
    #
    # model_name = dict()
    # model_name["ResNet"] = os.path.join(paths.models_path, resnet_name)
    # model_name["CNN-LSTM"] = os.path.join(paths.models_path, cnn_lstm_name)
    # cnn_architecture = "cnn_concat_lstm" if "cnn_concat_lstm" in cnn_lstm_name else "cnn_feed_lstm"
    #
    # feature_filename = "features_" + base_name + ".pkl"
    # save_feature_file(folder=parsed_train_folder, filename=feature_filename, model_name=model_name,
    #                   cnn_architecture=cnn_architecture, data_balance=True, **feature_type)
    #
    #
    # ### Train the classifier
    # classifier_name = "xgboost_" + base_name + ".dat"
    # classification.training(filename=feature_filename, model_name=classifier_name)






