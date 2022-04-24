import os
import time
import logging

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import paths
import utils
from read_data import read_expanded_data

from network import residual_network as rn
import dataloader as dl
import augmentation as aug


models_path = paths.models_path
figures_path = paths.figures_path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def training():

    # Parameter setting
    model_name = "resnet_" + time.strftime("%m-%d-%H:%M:%S") + ".pkl"
    initial_path = "model/resnet_06-14-01:19:07.pkl"
    num_epochs = 200
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 1e-3
    augmentation = True
    model_save_path = os.path.join(models_path, model_name)

    # initialize the network
    resnet_params = dict()
    resnet_params['num_features'] = 64
    resnet_params['in_channels'] = 1
    resnet_params['base_filters'] = 64
    resnet_params['kernel_size'] = 16
    resnet_params['stride'] = 2
    resnet_params['groups'] = 32
    resnet_params['n_block'] = 48
    resnet_params['n_classes'] = 4
    resnet_params['downsample_gap'] = 6
    resnet_params['increasefilter_gap'] = 12
    model = rn.load_network(device, path=initial_path, **resnet_params)
    model = model.to(device)
    model.verbose = False

    # Logging
    handler = logging.FileHandler("logging/resnet_" + time.strftime("%m-%d-%H:%M:%S") + ".txt")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("===================== Training params =====================")
    logger.info("Model name: " + model_name)
    logger.info("Initial path: " + str(initial_path))
    logger.info("Num of epochs: " + str(num_epochs))
    logger.info("Batch size: " + str(batch_size))
    logger.info("Learning rate: " + str(learning_rate))
    logger.info("Weight decay: " + str(weight_decay))
    logger.info("Data augmentation: " + str(augmentation))
    logger.info("================== Network architecture ===================")
    logger.info("Num of features: " + str(resnet_params['num_features']))
    logger.info("in channels: " + str(resnet_params['in_channels']))
    logger.info("base filters: " + str(resnet_params['base_filters']))
    logger.info("Kernel size: " + str(resnet_params['kernel_size']))
    logger.info("Stride: " + str(resnet_params['stride']))
    logger.info("Groups: " + str(resnet_params['groups']))
    logger.info("Num of blocks: " + str(resnet_params['n_block']))
    logger.info("Downsample gap: " + str(resnet_params['downsample_gap']))
    logger.info("increasefilter gap: " + str(resnet_params['increasefilter_gap']))

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = nn.CrossEntropyLoss()
    loss_func.to(device)

    transforms = None
    if augmentation:
        aug_params = dict()
        aug_params['sigma'] = 0.1
        aug_params['shift_interval'] = 20
        transforms = aug.data_augmentation(aug_params)
        logger.info("=================== Data augmentation =====================")
        logger.info("Sigma: " + str(aug_params['sigma']))
        logger.info("Shift interval: " + str(aug_params['shift_interval']))
    logger.info("===========================================================")

    # load data
    X_train, Y_train, _, = read_expanded_data(folder=paths.parsed_train_folder, mode="entire")
    X_test, Y_test, Name_list = read_expanded_data(folder=paths.parsed_test_folder, mode="entire")
    train_loader = dl.SupervisedLoader(X_train, Y_train, transforms=transforms)
    test_loader = dl.SupervisedLoader(X_test, Y_test)
    train_dataloader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_dataloader = DataLoader(test_loader, batch_size=batch_size, num_workers=4)

    # Training starts here
    min_test_loss = 10000
    train_history = []
    test_history = []
    f1_history = []
    training_size = len(train_dataloader.dataset)
    test_size = len(test_dataloader.dataset)
    print("Training size: ", training_size)
    print("Test size: ", test_size)

    initial_training_loss = 0.0
    initial_test_loss = 0.0
    for _, minibatch in enumerate(train_dataloader):
        with torch.set_grad_enabled(False):
            minibatch_x, minibatch_y = tuple(data.to(device) for data in minibatch)
            output = model(minibatch_x)
            loss = loss_func(output, minibatch_y)
            initial_training_loss += loss.item()

    for _, minibatch in enumerate(test_dataloader):
        with torch.set_grad_enabled(False):
            minibatch_x, minibatch_y = tuple(data.to(device) for data in minibatch)
            output = model(minibatch_x)
            loss = loss_func(output, minibatch_y)
            initial_test_loss += loss.item()
    print("Initial training loss: %.4f" % (initial_training_loss / training_size * batch_size))
    print("Initial test loss: %.4f" % (initial_test_loss / test_size * batch_size))
    print("-----------------------------------------------------------------")

    logger.info("Training size: " + str(training_size))
    logger.info("Test size: " + str(test_size))
    logger.info("Initial training loss: %.4f" % (initial_training_loss / training_size * batch_size))
    logger.info("Initial test loss: %.4f" % (initial_test_loss / test_size * batch_size))
    logger.info("===========================================================")

    early_stop = 0
    for epoch in range(num_epochs):
        bet = time.time()
        print("Current epoch: ", str(epoch + 1) + "/" + str(num_epochs))
        logger.info("Current epoch: " + str(epoch + 1))

        # Training
        train_running_loss = 0.0
        model.train()
        for _, minibatch in enumerate(train_dataloader):
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                minibatch_x, minibatch_y = tuple(data.to(device) for data in minibatch)
                output = model(minibatch_x)
                loss = loss_func(output, minibatch_y)
                loss.backward()
                optimizer.step()
                train_running_loss += loss.item()

        train_history.append(train_running_loss / training_size * batch_size)
        print("Train Loss: %.4f" % (train_running_loss / training_size * batch_size))
        logger.info("Train Loss: %.4f" % (train_running_loss / training_size * batch_size))

        # Validation
        test_running_loss = 0.0
        model.eval()
        all_output_prob = []
        for _, minibatch in enumerate(test_dataloader):
            with torch.set_grad_enabled(False):
                minibatch_x, minibatch_y = tuple(data.to(device) for data in minibatch)
                output = model(minibatch_x)
                all_output_prob.append(output.cpu().data.numpy())

                loss = loss_func(output, minibatch_y)
                test_running_loss += loss.item()

        all_output_prob = np.concatenate(all_output_prob)
        all_output = np.argmax(all_output_prob, axis=1)

        # vote most common
        final_pred = []
        final_true = []
        for name in np.unique(Name_list):
            temp_pred = all_output[Name_list == name]
            temp_true = Y_test[Name_list == name]
            final_pred.append(Counter(temp_pred).most_common(1)[0][0])
            final_true.append(Counter(temp_true).most_common(1)[0][0])

        f1_score, classification_report = utils.evaluation(final_true, final_pred, report=True)

        test_history.append(test_running_loss / test_size * batch_size)
        f1_history.append(f1_score)
        print("Test Loss: %.4f, " % (test_running_loss / test_size * batch_size), "Test F1: %.2f" % f1_score)
        print(classification_report)
        logger.info("Test Loss: %.4f, " % (test_running_loss / test_size * batch_size) + "Test F1: %.2f" % f1_score)
        logger.info(classification_report)

        scheduler.step(epoch)

        eet = time.time()
        if epoch == 0:
            print("Epoch time: %.2f" % (eet - bet), "seconds.")
            print("Estimated time to end: %.2f" % ((eet - bet) * (num_epochs - epoch)), "seconds.")
        print("-----------------------------------------------------------------")

        early_stop += 1
        # save the trained network
        if (test_running_loss / test_size * batch_size) < min_test_loss:
            torch.save(model.state_dict(), model_save_path)
            min_test_loss = test_running_loss / test_size * batch_size
            early_stop = 0
            print(">>> Model has been saved.")
            logger.info(">>> Model has been saved.")
        logger.info("-----------------------------------------------------------")

        if early_stop == 10:
          break

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(train_history, "r-")
    plt.plot(test_history, "b-")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.subplot(2, 1, 2)
    plt.plot(f1_history)
    plt.xlabel("Epoch")
    plt.ylabel("Test F1")
    plt.savefig(os.path.join(figures_path, model_name + ".png"), bbox_inches='tight', pad_inches=0)
    plt.show()

    return


def get_resnet_feature(data=None, names=None, name_list=None, model_name=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize the network
    resnet_params = dict()
    resnet_params['num_features'] = 64
    resnet_params['in_channels'] = 1
    resnet_params['base_filters'] = 64
    resnet_params['kernel_size'] = 16
    resnet_params['stride'] = 2
    resnet_params['groups'] = 32
    resnet_params['n_block'] = 48
    resnet_params['n_classes'] = 4
    resnet_params['downsample_gap'] = 6
    resnet_params['increasefilter_gap'] = 12
    model = rn.load_network(device, path=model_name, **resnet_params)
    model = model.to(device)

    print("(1) Predicting the features using the pre-trained ResNet...")
    temp_features = []
    num_of_data = len(data)
    minibatch_data = []
    for i in range(num_of_data):
        minibatch_data.append(data[i])
        if (i % 64 == 0 or i == (num_of_data - 1)) and i != 0:
            if i % 4096 == 0:
                print(">>> Current progress：", i,  "/", num_of_data)
            temp_input = torch.from_numpy(np.array(minibatch_data, dtype=np.float32)).type(torch.FloatTensor)
            temp_input = temp_input.to(device)
            temp_feature = model.get_features(temp_input)
            temp_features.extend(temp_feature.cpu().detach().numpy())
            minibatch_data = []

    temp_features = np.array(temp_features)

    print("(2) Combining the features extracted from the same signal...")
    num_of_y = len(name_list)
    features = [[0. for j in range(32)] for i in range(num_of_y)]  # (num_of_y, 32)

    # combine the features extracted from the same signal
    for i in range(len(temp_features)):
        pred_feature = np.array(temp_features[i], dtype=np.float32)
        current_name = str(names[i], 'utf-8')
        list_id = name_list[current_name]

        feature = np.array(features[list_id], dtype=np.float32)

        max_feature = 0
        for j in range(len(feature)):
            if feature[j] > max_feature:
                max_feature = feature[j]

        # if all values of the feature are less than 0, omit this feature
        if max_feature > 0:
            pred_feature = (pred_feature + feature) / 2

        features[list_id] = pred_feature

    resnet_features = []
    for i in range(len(features)):
        resnet_features.append(features[i])

    resnet_features = np.array(resnet_features)
    return resnet_features


def get_resnet_predition(data=None, names=None, name_list=None, model_name=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize the network
    resnet_params = dict()
    resnet_params['num_features'] = 64
    resnet_params['in_channels'] = 1
    resnet_params['base_filters'] = 64
    resnet_params['kernel_size'] = 16
    resnet_params['stride'] = 2
    resnet_params['groups'] = 32
    resnet_params['n_block'] = 48
    resnet_params['n_classes'] = 4
    resnet_params['downsample_gap'] = 6
    resnet_params['increasefilter_gap'] = 12
    model = rn.load_network(device, path=model_name, **resnet_params)
    model = model.to(device)

    print("(1) Predicting the predictions using the pre-trained ResNet...")
    temp_predictions = []
    num_of_data = len(data)
    minibatch_data = []
    for i in range(num_of_data):
        minibatch_data.append(data[i])
        if (i % 64 == 0 or i == (num_of_data - 1)) and i != 0:
            if i % 4096 == 0:
                print(">>> Current progress：", i,  "/", num_of_data)
            temp_input = torch.from_numpy(np.array(minibatch_data, dtype=np.float32)).type(torch.FloatTensor)
            temp_input = temp_input.to(device)
            temp_prediction = model(temp_input)
            temp_predictions.extend(temp_prediction.cpu().detach().numpy())
            minibatch_data = []

    temp_predictions = np.array(temp_predictions)

    print("(2) Combine the predictions extracted from the same signal...")
    num_of_y = len(name_list)
    predictions = [[0. for j in range(4)] for i in range(num_of_y)]  # (num_of_y, 4)

    # combine the predictions extracted from the same signal
    for i in range(len(temp_predictions)):
        pred_prediction = np.array(temp_predictions[i], dtype=np.float32)
        current_name = str(names[i], 'utf-8')
        list_id = name_list[current_name]

        prediction = np.array(predictions[list_id], dtype=np.float32)
        pred_prediction = (pred_prediction + prediction) / 2
        predictions[list_id] = pred_prediction

    resnet_predictions = []
    for i in range(len(predictions)):
        resnet_predictions.append(predictions[i])

    resnet_predictions = np.array(resnet_predictions)
    return resnet_predictions


if __name__ == '__main__':
    training()