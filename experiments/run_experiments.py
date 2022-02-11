import numpy as np
import random
import pandas as pd
import copy
import torch

import datasets.sim as sim
import datasets.mimic.mimic as mimic
import models.helpers as helpers
import models.R_models.r_models as r_models
import models.rmsn as rmsn
import models.crn as crn
import models.gnet as gnetfile
import models.deepace as deepace
import models.lstm_fact as lstm_fact
import models.static_methods as static
import yaml
import plotting.plots as plotting
import joblib
from sklearn.model_selection import train_test_split



def perform_experiments(run_config, load_hyper=True, method_params=None, return_results=False):
    # Set seed for reproducible results
    set_seeds(run_config["seed"])
    #Hyperparameters
    if load_hyper:
        method_params = load_hyperparams(run_config)
    methods = method_params.keys()
    number_exp = run_config["number_exp"]
    df_ates = pd.DataFrame(columns=methods)
    df_err = pd.DataFrame(columns=methods)

    for i in range(number_exp):
        print(f'Experiment {i}')

        #Estimate ACEs
        # Data
        d_train_seq, d_train_seq_unscaled, ates, a_int_1, a_int_2, y_scaler = generate_data(run_config, plot=False)
        print(f"ATE is {ates[-1]}")
        for method in methods:
            ace = estimate_ace(d_train_seq, d_train_seq_unscaled, a_int_1, a_int_2, method, method_params[method], run_config, y_scaler)
            err_ace = np.absolute(ace - ates[-1])
            print(method + f" ACE Error: {err_ace}")
            df_ates.loc[i, method] = ace
            df_err.loc[i, method] = err_ace
        print(f"Errors Experiment {i}:")
        print(df_err)

    # Save results in dataframe
    df_results = pd.DataFrame(columns=["method", "err", "sd"])
    df_results.iloc[:, 0] = methods
    df_results.iloc[:, 1] = df_err.mean(axis=0).to_numpy()
    df_results.iloc[:, 2] = df_err.std(axis=0).to_numpy()
    print(f"Results")
    print(df_results)

    #Return results
    if return_results:
        return df_results


#Seeds
def set_seeds(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#Intervention + data simulation
def generate_data(config, plot = False):
    T = config["T"]
    # Intervention sequences
    a_int_1, a_int_2 = sim.generate_interventions(T, config["treat"], config["dataset"])

    # Generate data
    if config["dataset"] == "sim":
        d_seq_unscaled, d_seq_cf_unscaled1, d_seq_cf_unscaled2 = sim.simulate_data(config, a_int_1, a_int_2)
        ates = helpers.estimate_ates_from_cf(d_seq_cf_unscaled1, d_seq_cf_unscaled2, T)
    if config["dataset"] == "mimic":
        d_seq_unscaled, d_seq_cf_unscaled1, d_seq_cf_unscaled2 = mimic.generate_data(config, a_int_1, a_int_2)
        ates = helpers.estimate_ates_from_cf(d_seq_cf_unscaled1, d_seq_cf_unscaled2, T)
    # Standardize
    d_train_seq, y_scaler = helpers.standardize_data(copy.deepcopy(d_seq_unscaled))
    # Plot data
    if plot:
        plotting.plot_data(d_seq_unscaled, d_seq_cf_unscaled1, d_seq_cf_unscaled2, observation_id=5, cov_id=2)
    return d_train_seq, d_seq_unscaled, ates, a_int_1, a_int_2, y_scaler





#Hyperparameter loading for all methods
def load_hyperparams(run_config):
    # Methods to run
    methods = run_config["methods"]
    # Dictionary consisting of method names and associates hyperparameters
    method_params = {}
    # Hyperparameters
    save_dir_hyper = helpers.get_project_path() + "/hyperparam/parameters/" + run_config["dataset"] + "/"
    for method in methods:
        if method in ["rmsn", "crn", "gnet", "deepace", "deepace_tar", "lstm_fact", "dragonnet", "tarnet"]:
            #Check whether to use tuned hyperparameters
            if run_config["load_hyperparam"]:
                if method == 'rmsn':
                    params_propnom = joblib.load(save_dir_hyper + "study_rmsn_prop_nom.pkl").best_trial.params
                    params_propdenom = joblib.load(save_dir_hyper + "study_rmsn_prop_nom.pkl").best_trial.params
                    params_enc = joblib.load(save_dir_hyper + "study_rmsn_enc.pkl").best_trial.params
                    params_dec = joblib.load(save_dir_hyper + "study_rmsn_dec.pkl").best_trial.params
                    method_params[method] = [params_propnom, params_propdenom, params_enc, params_dec]
                elif method == "crn":
                    params_enc = joblib.load(save_dir_hyper + "study_crn_enc.pkl").best_trial.params
                    params_dec = joblib.load(save_dir_hyper + "study_crn_dec.pkl").best_trial.params
                    method_params[method] = [params_enc, params_dec]
                elif method == "deepace":
                    study_name = "study_deepace_tar_treat" + str(run_config["treat"]) + "_int"
                    params_1 = joblib.load(save_dir_hyper + "deepace/" + study_name + "1.pkl").best_trial.params
                    params_2 = joblib.load(save_dir_hyper + "deepace/" + study_name + "2.pkl").best_trial.params
                    method_params[method] = [params_1, params_2]
                elif method == "deepace_tar":
                    study_name = "study_deepace_tar_treat" + str(run_config["treat"]) + "_int"
                    params_1 = joblib.load(save_dir_hyper + "deepace/" + study_name + "1.pkl").best_trial.params
                    params_2 = joblib.load(save_dir_hyper + "deepace/" + study_name + "2.pkl").best_trial.params
                    method_params[method] = [params_1, params_2]
                else:
                    method_params[method] = joblib.load(save_dir_hyper + "study_" + method + ".pkl").best_trial.params
            else:
                # Same fixed hyperparameters for every method
                params_static = {'hidden_size_lstm': 10, 'hidden_size_lstm2': 10, 'hidden_size_body': 10,
                                 'hidden_size_head': 10,
                                 'lr': 5e-4, 'dropout': 0, 'batch_size': 100}
                if method == 'rmsn':
                    method_params[method] = [params_static, params_static, params_static, params_static]
                elif method in ["crn", "deepace", "deepace_tar"]:
                    method_params[method] = [params_static, params_static]
                else:
                    method_params[method] = params_static
        else:
            method_params[method] = None

    return method_params


#ACE estimation wrapper depending on method
def estimate_ace(data_scaled, data_unscaled, a_int_1, a_int_2, method, hyperparams, run_config, y_scaler = None):
    # Configuration parameters
    p = run_config["p"]
    ate = None
    if method=="deepace":
        # Model training
        deepace1, _ = deepace.train_deepace(hyperparams[0], data_scaled, a_int_1, alpha=0, beta=0, tune_mode=False)
        deepace2, _ = deepace.train_deepace(hyperparams[1], data_scaled, a_int_2, alpha=0, beta=0, tune_mode=False)
        ate = deepace1.estimate_avg_outcome(data_scaled, y_scaler) - deepace2.estimate_avg_outcome(data_scaled, y_scaler)
    if method == "deepace_tar":
        # Model training
        deepace1, _ = deepace.train_deepace(hyperparams[0], data_scaled, a_int_1, alpha=0.1, beta=0.01, tune_mode=False)
        deepace2, _ = deepace.train_deepace(hyperparams[1], data_scaled, a_int_2, alpha=0.1, beta=0.01, tune_mode=False)
        ate = deepace1.estimate_avg_outcome(data_scaled, y_scaler) - deepace2.estimate_avg_outcome(data_scaled, y_scaler)
    if method in ["ltmle", "gcomp", "gcomp_par", "ltmle_super", "msm", "snm"]:
        ate = r_models.run_r_model(data_unscaled, p + 3, method, a_int_1, a_int_2)
    if method=="crn":
        enc_crn, dec_crn = crn.train_enc_dec_joint(hyperparams[0], hyperparams[1], data_scaled)
        ate = dec_crn.estimate_ace(data_scaled, a_int_1, a_int_2, y_scaler=y_scaler)
    if method=="rmsn":
        enc_rmsn, dec_rmsn = rmsn.train_rmsn_joint(hyperparams[0], hyperparams[1], hyperparams[2], hyperparams[3], data_scaled)
        ate = dec_rmsn.estimate_ace(data_scaled, a_int_1, a_int_2, y_scaler=y_scaler)
    if method=="gnet":
        d_train, d_holdout = train_test_split(data_scaled, test_size=0.2, shuffle=False)
        gnet, _ = gnetfile.train_gnet(hyperparams, d_train, n_cov_cont= run_config["n_cov_cont"]+1, n_cov_discr=run_config["n_cov_discr"])
        ate = gnet.estimate_ace(d_train, d_holdout, a_int_1, a_int_2, K=100, y_scaler=y_scaler)
    if method=="lstm_fact":
        lstm_f, _ = lstm_fact.train_lstm_fact(hyperparams, data_scaled)
        ate = lstm_f.estimate_ace(data_scaled, a_int_1, a_int_2, y_scaler=y_scaler)
    if method=="dragonnet":
        ate = static.ace_dragonnet(hyperparams, data_scaled, a_int_1, a_int_2, alpha=1, beta=1, y_scaler=y_scaler)
    if method == "tarnet":
        ate = static.ace_dragonnet(hyperparams, data_scaled, a_int_1, a_int_2, alpha=0, beta=0, y_scaler=y_scaler)
    if method in ["dml_forest", "dml_kernel", "static"]:
        ate = static.ace_econml(data_scaled, a_int_1, a_int_2, method, y_scaler=y_scaler)

    if ate is None:
        raise ValueError(f'Method {method} wasnt recognized')
    return ate


if __name__ == "__main__":
    # Coniguration for run
    stream = open(helpers.get_project_path() + "/experiments/config/sim/config_deepace_tar.yaml", 'r')
    run_config = yaml.safe_load(stream)
    perform_experiments(run_config)
