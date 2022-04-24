import csv
import numpy as np


def read_long_data(folder=None):
    long_signals = list()
    long_labels = list()
    long_names = list()
    try:
        with open(folder + 'long.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                long_signals.append(list(map(float, row[2:])))
                long_labels.append(row[1])
                long_names.append(row[0])
        print("{0} long data is loaded.".format(len(long_names)))
        return long_signals, long_labels, long_names
    except:
        raise FileExistsError("The current filename do not exist.")


def read_qrs_data(folder=None):
    qrs_infos = list()
    qrs_labels = list()
    qrs_names = list()
    try:
        with open(folder + 'qrs_info.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                qrs_infos.append(list(map(float, row[2:])))
                qrs_labels.append(row[1])
                qrs_names.append(row[0])
        print("{0} qrs data is loaded.".format(len(qrs_names)))
        return qrs_infos, qrs_labels, qrs_names
    except:
        raise FileExistsError("The current filename do not exist.")


def read_short_data(folder=None):
    short_signals = list()
    short_labels = list()
    short_names = list()
    try:
        with open(folder + 'short.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                short_signals.append(list(map(float, row[2:])))
                short_labels.append(row[1])
                short_names.append(row[0])
        print("{0} short data is loaded.".format(len(short_names)))
        return short_signals, short_labels, short_names
    except:
        raise FileExistsError("The current filename do not exist.")


def read_extracted_short_data(folder=None, mode="entire"):
    try:
        if mode == "two_part":
            with open(folder + "extracted_data_2part.bin", 'rb') as file:
                train_signal = np.load(file)
                train_label = np.load(file)
                train_name = np.load(file)
                test_signal = np.load(file)
                test_label = np.load(file)
                test_name = np.load(file)

            labels = list(train_label)
            num_samples = len(labels)
            class_weights = np.log2(
                [num_samples / labels.count(0), num_samples / labels.count(1),
                 num_samples / labels.count(2), num_samples / labels.count(3)]
            )

            print("{0} extracted train data is loaded.".format(train_name.shape[0]))
            print("{0} extracted test data is loaded.".format(test_name.shape[0]))
            return train_signal, train_label, train_name, test_signal, test_label, test_name, class_weights

        else:
            with open(folder + "extracted_short.bin", 'rb') as file:
                extracted_short_signals = np.load(file, allow_pickle=True)
                extracted_short_labels = np.load(file, allow_pickle=True)
                extracted_short_names = np.load(file, allow_pickle=True)

            labels = list(extracted_short_labels)
            num_samples = len(labels)
            class_weights = np.log2(
                [num_samples / labels.count(0), num_samples / labels.count(1),
                 num_samples / labels.count(2), num_samples / labels.count(3)]
            )
            print("{0} extracted data is loaded.".format(extracted_short_names.shape[0]))
            return extracted_short_signals, extracted_short_labels, extracted_short_names, class_weights

    except:
        raise FileExistsError("The current filename do not exist.")


def read_expanded_data(folder=None, mode="entire"):
    try:
        if mode == "two_part":
            with open(folder + "expanded_data_2part.bin", 'rb') as file:
                train_signal = np.load(file)
                train_label = np.load(file)
                train_name = np.load(file)
                test_signal = np.load(file)
                test_label = np.load(file)
                test_name = np.load(file)
            print("{0} expanded train data is loaded.".format(train_name.shape[0]))
            print("{0} expanded test data is loaded.".format(test_name.shape[0]))
            return train_signal, train_label, train_name, test_signal, test_label, test_name

        else:
            with open(folder + "expanded_data.bin", 'rb') as file:
                expanded_signals = np.load(file)
                expanded_labels = np.load(file)
                expanded_names = np.load(file)
            print("{0} expanded data is loaded.".format(expanded_names.shape[0]))
            return expanded_signals, expanded_labels, expanded_names
    except:
        raise FileExistsError("The current filename do not exist.")


def read_balanced_data(folder=None, datatype='long'):
    balanced_signals = list()
    balanced_labels = list()
    balanced_names = list()
    try:
        with open(folder + "balanced_" + datatype + ".csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                balanced_signals.append(list(map(float, row[2:])))
                balanced_labels.append(row[1])
                balanced_names.append(row[0])
        print("{0} balanced ".format(len(balanced_names)) + datatype + " data is loaded.")
        return balanced_signals, balanced_labels, balanced_names
    except:
        raise FileExistsError("The current filename do not exist.")


if __name__ == "__main__":
    pass