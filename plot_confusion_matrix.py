import os
import time

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import paths


sns.set(
    context="notebook",
    style="whitegrid",
    rc={"figure.dpi": 120, "scatter.edgecolors": "k"},
)

figures_path = paths.figures_path

df_pred = pd.read_csv("PREDICTIONS.csv", header=None).values  # Klassifikationen
df_gt = pd.read_csv(paths.test_csv_path, header=None).values  # Wahrheit

y_pred = []
y_gt = []
for [name, label] in df_pred:
    y_pred.append(label)
for [name, label] in df_gt:
    y_gt.append(label)

y_pred = np.array(y_pred)
y_gt = np.array(y_gt)


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray):
    """Plots the confusion matrix given the true test labels y_test and the predicted labes y_pred"""
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)  # ndarray of shape (n_classes, n_classes)
    df_cm = pd.DataFrame(cm)
    sns.heatmap(df_cm, annot=True, fmt='.20g', cmap=plt.cm.Greens)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.grid(False)
    plt.savefig(os.path.join(figures_path, time.strftime("%m-%d-%H:%M:%S") + ".png"))
    plt.show()


plot_confusion_matrix(y_gt, y_pred)