import pandas as pd
import numpy as np
import models.helpers as helpers


def load_data():
    path = helpers.get_project_path() + "/datasets/back_pain/data_preprocessed/"
    f = open(path + "baseline_names.txt", "r")
    baseline_names = str.split(f.read(), "\n")
    f = open(path + "subgroup_names.txt", "r")
    subgroup_names = str.split(f.read(), "\n")

    #Load data and inpute missing values
    outcomes = pd.read_csv(path + "outcomes.csv")
    outcomes = outcomes.fillna(method='ffill')
    outcomes = outcomes.fillna(method='bfill')
    baselines = pd.read_csv(path + "baselines.csv", encoding = "ISO-8859-1")
    baselines = baselines.fillna(method='ffill')
    baselines = baselines.fillna(method='bfill')
    data_lbp = pd.read_csv(path + "lbp.csv", encoding = "ISO-8859-1")
    data_lbp = data_lbp.fillna(method='ffill')
    data_lbp = data_lbp.fillna(method='bfill')

    n = data_lbp.shape[0]
    T = 3

    #Subgroups
    subgroups = data_lbp["pp_single_modal"]
    groups = np.zeros(n)
    for i in range(n):
        groups[i] = subgroups[i][2]


    #Create outcomes tensor
    out = np.zeros((n, T, 2))
    for t in range(T):
        out[:, t, 0] = outcomes.iloc[:, t]
        out[:, t, 1] = outcomes.iloc[:, t+T]
    #Scale outcomes
    out_unscaled = out.copy()
    out[:, :, 0], y_scaler1 = helpers.standardize_outcome(out[:, :, 0])
    out[:, :, 1], y_scaler2 = helpers.standardize_outcome(out[:, :, 1])

    #Treatment
    treat = baselines["bfbe0"].to_numpy()
    for i in range(n):
        if treat[i] > 1:
            treat[i] = 1
        else:
            treat[i] = 0
    A = np.zeros((n, T, 1))
    for t in range(T):
        A[:, t, 0] = treat

    #Y
    y = out[:, :, 1:2]
    y_us = out_unscaled[:, :, 1:2]

    #Static covariates
    baselines_processed = process_baselines(baselines.drop("bfbe0", axis=1))
    x_static = np.zeros((n, T, baselines_processed.shape[1]))
    for t in range(T):
        x_static[:, t, :] = baselines_processed.iloc[:, :]

    #Covariates
    y_lagged = np.zeros((n, T, 1))
    y_lagged[:, 1:, 0] = y[:, 0:(T-1), 0]
    x = np.concatenate((out[:, :, 0:1], y_lagged, x_static), axis=2)


    #Concatinate
    data = np.concatenate((y, A, x), axis=2)
    data_unscaled = np.concatenate((y_us, A, x), axis=2)


    return data, data_unscaled, groups, y_scaler1


def process_baselines(baselines):
    processed_static_features = []
    for feature in baselines.columns:
        if isinstance(baselines[feature].iloc[0], float):
            mean = np.mean(baselines[feature])
            std = np.std(baselines[feature])
            processed_static_features.append((baselines[feature] - mean) / std)
        elif isinstance(baselines[feature].iloc[0], np.int64):
            processed_static_features.append(baselines[feature].astype("float"))
        else:
            one_hot = pd.get_dummies(baselines[feature])
            processed_static_features.append(one_hot.astype(float))

    static_features = pd.concat(processed_static_features, axis=1)
    return static_features