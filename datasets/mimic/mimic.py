import pandas as pd
import numpy as np
import models.helpers as helpers
from scipy.special import expit


def simulate_treat_outcomes(config, a_int_1, a_int_2, data_vitals):
    # Treatment assignment
    n = config["n"]
    T = config["T"]
    h = config["lag"]
    # Arrays to store treatments/ outcomes
    A_f = np.zeros((n, T))
    A_cf1 = np.tile(np.expand_dims(a_int_1, 0), (n, 1))
    A_cf2 = np.tile(np.expand_dims(a_int_2, 0), (n, 1))
    Y_f = np.zeros((n, T))
    Y_cf1 = np.zeros((n, T))
    Y_cf2 = np.zeros((n, T))
    print(np.mean(data_vitals))
    # Noise
    def noise(s, sd):
        return np.random.normal(loc=0, scale=sd, size=s)

    err_A = noise((n, T), config["noise_A"])
    err_Y = noise((n, T), config["noise_Y"])

    # Coefficients
    coef_xa = np.empty(h)
    coef_ya = np.empty(h)
    coef_xy = np.empty(h)
    coef_ay = np.empty(h)
    coef_yy = np.empty(h)
    for i in range(h):
        coef_xa[i] = ((-1) ** i) * (1 / (i + 1))
        coef_ya[i] = ((-1) ** i) * (1 / (i + 1))
        coef_xy[i] = ((-1) ** i) * (1 / (i + 1))
        coef_ay[i] = ((-1) ** i) * (1 / (i + 1))
        coef_yy[i] = ((-1) ** i) * (1 / (i + 1))

    # Data generation
    def generate_data(A=None, cf=False):
        if A is None:
            A = np.zeros((n, T))
        Y = np.zeros((n, T))
        treat_level_mid = T / 2
        treat_level_max = T
        treat_level = np.full(n, treat_level_mid - 3)
        for t in range(0, T):
            t_start = max(0, t - h)
            hist = data_vitals[:, t_start:(t + 1), :]
            hist_y = np.tanh(Y[:, t_start:t]/2)
            hist_mean = np.mean(hist, axis=2)
            if not cf:
                # Treatment assignment
                # previous covariates + outcomes
                a_contrib = err_A[:, t] - np.tanh(treat_level - treat_level_mid)
                for i in range(min(h, t + 1)):
                    a_contrib += coef_xa[i] * hist_mean[:, hist_mean.shape[1] - 1 - i]
                for i in range(min(h - 1, t)):
                    a_contrib += coef_ya[i] * hist_y[:, hist_y.shape[1] - 1 - i]
                prob_A = expit(a_contrib)
                A[:, t] = np.where(prob_A > 0.5, 1, 0)

            # Adjust patient medication level
            if t > 1:
                treat_level += (2 * A[:, t] - np.ones(n))* np.abs(np.mean(data_vitals[:, t, :], axis=1)*np.tanh(Y[:, t-1]))
            else:
                treat_level += (2 * A[:, t] - np.ones(n)) * np.abs(np.mean(data_vitals[:, t, :], axis=1))
            if t > 0:
                treat_level += (2 * A[:, t - 1] - np.ones(n))
            for i in range(n):
                if treat_level[i] < 0:
                    treat_level[i] = 0
                if treat_level[i] > treat_level_max:
                    treat_level[i] = treat_level_max

            # Outcomes
            base_term = err_Y[:, t]
            Y[:, t] = base_term
            past_term = 0
            for i in range(h):
                if t-i >= 0:
                    past_term += coef_ya[i]* np.tanh(np.sin(np.mean(data_vitals[:, t-i, 0:5], axis=1)*t)*A[:, t-i] +
                                       np.cos(np.mean(data_vitals[:, t-i, 5:], axis=1)*t)*A[:, t-i])
            Y[:, t] = 5*past_term + base_term
        return A, Y

    A_f, Y_f = generate_data(cf=False)
    A_cf1, Y_cf1 = generate_data(A_cf1, cf=True)
    A_cf2, Y_cf2 = generate_data(A_cf2, cf=True)

    def create_dataset(Y, A):
        data = np.concatenate((np.expand_dims(Y, 2), np.expand_dims(A, 2), data_vitals), axis=2)
        # Add lagged outcomes
        Y_prev = Y.copy()
        Y_prev[:, 0] = np.zeros(n)
        Y_prev[:, 1:] = Y[:, 0:(T - 1)]
        data = np.concatenate((data, np.expand_dims(Y_prev, 2)), 2)
        return data

    return create_dataset(Y_f, A_f), create_dataset(Y_cf1, A_cf1), create_dataset(Y_cf2, A_cf2)


def load_mimic_covariates(n, T, vital_list, static_list):
    data_path = helpers.get_project_path() + "/datasets/mimic/all_hourly_data.h5"
    h5 = pd.HDFStore(data_path, 'r')

    all_vitals = h5['/vitals_labs_mean'][vital_list]

    all_vitals = all_vitals.droplevel(['hadm_id', 'icustay_id'])
    column_names = []
    for column in all_vitals.columns:
        if isinstance(column, str):
            column_names.append(column)
        else:
            column_names.append(column[0])
    all_vitals.columns = column_names

    # Filling NA
    all_vitals = all_vitals.fillna(method='ffill')
    all_vitals = all_vitals.fillna(method='bfill')

    # Static features
    if static_list is not None:
        static_features = h5['/patients'][static_list]
        static_features = static_features.droplevel(['hadm_id', 'icustay_id'])

    # Filtering out users with time length < T
    user_sizes = all_vitals.groupby('subject_id').size()
    filtered_users_len = user_sizes.index[user_sizes >= 2*T]

    # Filtering out users with time age > 100
    if static_list is not None:
        if "age" in static_list:
            filtered_users_age = static_features.index[static_features.age < 100]
            filtered_users = filtered_users_len.intersection(filtered_users_age)
        else:
            filtered_users = filtered_users_len
    else:
        filtered_users = filtered_users_len

    filtered_users = np.random.choice(filtered_users, size=n, replace=False)
    all_vitals = all_vitals.loc[filtered_users]
    # Cut off > T
    #all_vitals = all_vitals.groupby('subject_id').head(T)

    vitals_grouped = all_vitals.groupby('subject_id')
    data_vitals = np.zeros((n, T, len(vital_list)))
    for i, cov in enumerate(vitals_grouped):
        test = cov[1].to_numpy()
        for t in range(T):
            data_vitals[i, t, :] = test[t, :]

    # Standardize
    data_vitals = helpers.standardize_covariates(data_vitals)

    # One-hot encoding/ Standardization for static covariates
    if static_list is not None:
        static_features = static_features.loc[filtered_users]
        processed_static_features = []
        for feature in static_features.columns:
            if not isinstance(static_features[feature].iloc[0], float):
                one_hot = pd.get_dummies(static_features[feature])
                processed_static_features.append(one_hot.astype(float))
            else:
                mean = np.mean(static_features[feature])
                std = np.std(static_features[feature])
                processed_static_features.append((static_features[feature] - mean) / std)

        static_features = pd.concat(processed_static_features, axis=1).to_numpy()
    else:
        static_features = None

    return data_vitals, static_features
