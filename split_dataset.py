import pandas as pd

import paths

data = pd.read_csv(paths.original_csv_path, header=None).values
train_names, test_names = data[:5500, 0], data[5500:, 0]
train_labels, test_labels = data[:5500, 1], data[5500:, 1]

# save the training/test .csv file
to_save_train_csv = pd.DataFrame({'name': train_names,'label': train_labels})
to_save_test_csv = pd.DataFrame({'name': test_names,'label': test_labels})

to_save_train_csv.to_csv(paths.train_csv_path, sep=',', header=False, index=False)
to_save_test_csv.to_csv(paths.test_csv_path, sep=',', header=False, index=False)

