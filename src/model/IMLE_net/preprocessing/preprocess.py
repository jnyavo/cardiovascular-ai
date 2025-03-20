"""Preprocessing pipeline for the dataset.

This file can is imported as a module and contains the following functions:

    * apply_scaler - Applies standard scaler to an ECG signal.
    * preprocess - Preprocesses the dataset.

"""
__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"


import os
from typing import Tuple

import numpy as np
import pandas as pd
import wfdb, ast
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

def apply_scaler(inputs: np.array, scaler: StandardScaler) -> np.array:
    """Applies standardization to each individual ECG signal.

    Parameters
    ----------
    inputs: np.array
        Array of ECG signals.
    scaler: StandardScaler
        Standard scaler object.

    Returns
    -------
    np.array
        Array of standardized ECG signals.

    """

    temp = []
    for x in inputs:
        x_shape = x.shape
        temp.append(scaler.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    temp = np.array(temp)
    return temp


def preprocess(X_data: np.ndarray, y: np.ndarray, Y_data: pd.DataFrame, scaler_path: str = 'bin/standardization.bin'):
    """Preprocesses the dataset.

    Parameters
    ----------
    X_data: np.ndarray
        Array of ECG signals.
    y: np.ndarray
        Array of labels corresponding to the ECG signals.
    Y_data: pd.DataFrame
        DataFrame containing metadata for the ECG signals.

    Returns
    -------
    tuple[np.array]
        Tuple of arrays containing train, valid and test data.

    """

    print("Preprocessing dataset...", end="\n" * 2)


    # Stratified split
    X_train = X_data[Y_data.strat_fold < 9]
    y_train = y[Y_data.strat_fold < 9]

    X_val = X_data[Y_data.strat_fold == 9]
    y_val = y[Y_data.strat_fold == 9]

    X_test = X_data[Y_data.strat_fold == 10]
    y_test = y[Y_data.strat_fold == 10]

    del X_data, Y_data, y

    # Standardization
    scaler_path = os.path.join(os.getcwd(), scaler_path)
    scaler: StandardScaler
    if os.path.exists(scaler_path):
        scaler = load(scaler_path)
        print(f"Loading scaler from {scaler_path}")
    else:
        scaler = StandardScaler()
        scaler.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))
        dump(scaler, scaler_path)
    X_train_scale = apply_scaler(X_train, scaler)
    X_test_scale = apply_scaler(X_test, scaler)
    X_val_scale = apply_scaler(X_val, scaler)

    del X_train, X_test, X_val

    # Shuffling
    X_train_scale, y_train = shuffle(X_train_scale, y_train, random_state=42)

    return X_train_scale, y_train, X_test_scale, y_test, X_val_scale, y_val
