# -*- coding: utf-8 -*-
"""
Diese Datei sollte nicht verändert werden und wird von uns gestellt und zurückgesetzt.

Skript testet das vortrainierte Modell


@author: Maurice Rohr
"""

from predict import predict_labels
from wettbewerb import load_references, save_predictions
from score import score

import paths

if __name__ == '__main__':

    folder = paths.original_test_folder

    ecg_leads, ecg_labels, fs, ecg_names = load_references(folder=folder) # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz
    
    predictions = predict_labels(ecg_leads, fs, ecg_names, use_pretrained=True)
    
    save_predictions(predictions)  # speichert Prädiktion in CSV Datei

    F1, F1_mult = score()
    print("F1:", F1, "\t MultilabelScore:", F1_mult)





