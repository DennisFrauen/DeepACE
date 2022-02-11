import numpy as np
import random
import torch
import yaml
from optuna.samplers import TPESampler

import datasets.sim as sim
import datasets.mimic.mimic as mimic
import models.helpers as helpers
import models.rmsn as rmsn
import models.crn as crn
import models.deepace as deepace
import models.gnet as gnet
import models.lstm_fact as lstm_fact
import models.static_methods as static
import joblib
from sklearn.model_selection import train_test_split
import datasets.back_pain.load_backpain as backpain

def set_seeds(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tune_sampler = TPESampler(seed=seed)
    return tune_sampler

#DeepACE
def tune_deepace(data, a_int, study_name, alpha, beta, num_samples=10, path="", num_epochs=100, tune_sampler=None):
    def objective(trial):
        p = int((data.shape[2] - 2)/2)
        config = {
            "hidden_size_lstm": trial.suggest_categorical("hidden_size_lstm", [p, 2*p, 3*p, 4*p]),
            "hidden_size_body": trial.suggest_categorical("hidden_size_body", [p, 2*p, 3*p, 4*p]),
            "hidden_size_head": trial.suggest_categorical("hidden_size_head", [p, 2*p, 3*p, 4*p]),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
        }
        _, val_err = deepace.train_deepace(config, data, a_int=a_int, alpha=alpha, beta=beta, epochs=num_epochs, tune_mode=True)
        return val_err

    helpers.tune_hyperparam(objective, study_name, path, num_samples, tune_sampler)

#CRN--------------------------------------------------------------------------------
#Encoder
def tune_crn_enc(data, num_samples=10, path="", num_epochs=100, tune_sampler=None):
    def objective(trial):
        p = int((data.shape[2] - 2)/2)
        config = {
            "hidden_size_lstm": trial.suggest_categorical("hidden_size_lstm", [p, 2*p, 3*p, 4*p]),
            "hidden_size_body": trial.suggest_categorical("hidden_size_body", [p, 2*p, 3*p, 4*p]),
            "hidden_size_head": trial.suggest_categorical("hidden_size_head", [p, 2*p, 3*p, 4*p]),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
        }
        _, val_err = crn.train_Encoder_CRN(config, data, num_epochs, tune_mode=True)
        return val_err
    helpers.tune_hyperparam(objective, "study_crn_enc", path, num_samples, tune_sampler)

#Decoder
def tune_crn_dec(data, encoder, num_samples=10, path="", num_epochs=100, tune_sampler=None):
    def objective(trial):
        p = int((data.shape[2] - 2)/2)
        config = {
            "hidden_size_lstm": trial.suggest_categorical("hidden_size_lstm", [p, 2*p, 3*p, 4*p]),
            "hidden_size_body": trial.suggest_categorical("hidden_size_body", [p, 2*p, 3*p, 4*p]),
            "hidden_size_head": trial.suggest_categorical("hidden_size_head", [p, 2*p, 3*p, 4*p]),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
        }
        _, val_err = crn.train_Decoder_CRN(config, data, encoder, num_epochs, tune_mode=True)
        return val_err
    helpers.tune_hyperparam(objective, "study_crn_dec", path, num_samples, tune_sampler)

#RMSN------------------------------------------------------------------------------------------------
#Nominator network
def tune_param_RMSN_nom(data, num_samples=10, path="", num_epochs=100, tune_sampler=None):
    def objective_nom(trial):
        p = int((data.shape[2] - 2)/2)
        config = {
            "hidden_size_lstm": trial.suggest_categorical("hidden_size_lstm", [p, 2*p, 3*p, 4*p]),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
        }
        d_train, d_val = train_test_split(data, test_size=0.2, shuffle=True)
        propnet_nom = rmsn.train_nominator(config, d_train, num_epochs)
        val_err = propnet_nom.validation_step(d_val)
        return val_err
    helpers.tune_hyperparam(objective_nom, "study_rmsn_prop_nom", path, num_samples, tune_sampler)

# Denominator network
def tune_param_RMSN_denom(data, num_samples=10, path="", num_epochs=100, tune_sampler=None):
    def objective_denom(trial):
        p = int((data.shape[2] - 2)/2)
        config = {
            "hidden_size_lstm": trial.suggest_categorical("hidden_size_lstm", [p, 2*p, 3*p, 4*p]),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
        }
        d_train, d_val = train_test_split(data, test_size=0.2, shuffle=True)
        propnet_denom = rmsn.train_denominator(config, d_train, num_epochs)
        val_err = propnet_denom.validation_step(d_val)
        return val_err

    helpers.tune_hyperparam(objective_denom, "study_rmsn_prop_denom", path, num_samples, tune_sampler)

#Encoder network
def tune_param_RMSN_enc(data, SW, num_samples=10, path="", num_epochs=100, tune_sampler=None):
    def objective_enc(trial):
        p = int((data.shape[2] - 2)/2)
        config = {
            "hidden_size_lstm": trial.suggest_categorical("hidden_size_lstm", [p, 2*p, 3*p, 4*p]),
            "hidden_size": trial.suggest_categorical("hidden_size", [p, 2*p, 3*p, 4*p]),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
        }
        _, val_err = rmsn.train_rmsn_enc(config, data, SW, num_epochs, tune_mode=True)
        return val_err

    helpers.tune_hyperparam(objective_enc, "study_rmsn_enc", path, num_samples, tune_sampler)

#Decoder network
def tune_param_RMSN_dec(data, encoder, SW, num_samples=10, path="", num_epochs=100, tune_sampler=None):
    def objective_enc(trial):
        p = int((data.shape[2] - 2)/2)
        config = {
            "hidden_size_lstm": trial.suggest_categorical("hidden_size_lstm", [p, 2*p, 3*p, 4*p]),
            "hidden_size": trial.suggest_categorical("hidden_size", [p, 2*p, 3*p, 4*p]),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
        }
        _, val_err = rmsn.train_rmsn_dec(config, data, encoder, SW, num_epochs, tune_mode=True)
        return val_err

    helpers.tune_hyperparam(objective_enc, "study_rmsn_dec", path, num_samples, tune_sampler)


#GNet--------------------------------------------------------------------------------------------------
def tune_gnet(data, n_cov_cont, n_cov_discr, num_samples=10, path="", num_epochs=100, tune_sampler=None):
    def objective(trial):
        p = int((data.shape[2] - 2)/2)
        config = {
            "hidden_size_lstm": trial.suggest_categorical("hidden_size_lstm", [p, 2*p, 3*p, 4*p]),
            "hidden_size_lstm2": trial.suggest_categorical("hidden_size_lstm2", [p, 2 * p, 3 * p, 4 * p]),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
        }
        _, val_err = gnet.train_gnet(config, data, n_cov_cont, n_cov_discr, num_epochs, tune_mode=True)
        return val_err
    helpers.tune_hyperparam(objective, "study_gnet", path, num_samples, tune_sampler)


#Factual LSTM--------------------------------------------------------------------------------------------------
def tune_lstm_fact(data, num_samples=10, path="", num_epochs=100, tune_sampler=None):
    def objective(trial):
        p = int((data.shape[2] - 2)/2)
        config = {
            "hidden_size_lstm": trial.suggest_categorical("hidden_size_lstm", [p, 2*p, 3*p, 4*p]),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
        }
        _, val_err = lstm_fact.train_lstm_fact(config, data, num_epochs, tune_mode=True)
        return val_err
    helpers.tune_hyperparam(objective, "study_lstm_fact", path, num_samples, tune_sampler)


#Dragonnet--------------------------------------------------------------------------------------------------
def tune_dragonnet(data, num_samples=10, path="", num_epochs=100, tune_sampler=None, alpha=1, beta=1):
    def objective(trial):
        p = int((data.shape[2] - 2)/2)
        T = data.shape[1]
        s = ((T-1)*(p+1))+p
        config = {
            "hidden_size_body": trial.suggest_categorical("hidden_size_body", [int(s/2), s, 2 * s, 3 * s]),
            "hidden_size_head": trial.suggest_categorical("hidden_size_head", [int(s/2), s, 2 * s, 3 * s]),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
        }
        data_dn = static.format_data_drag(data)
        _, val_err = static.train_dragonnet(config, data_dn, alpha, beta, num_epochs)
        return val_err
    if alpha==1:
        helpers.tune_hyperparam(objective, "study_dragonnet", path, num_samples, tune_sampler)
    else:
        helpers.tune_hyperparam(objective, "study_tarnet", path, num_samples, tune_sampler)

def generate_data(hyper_config):
    # interventions
    T = hyper_config["T"]
    a_int_1, a_int_2 = sim.generate_interventions(T, hyper_config["treat"], hyper_config["dataset"])
    # Generate data
    d_train_seq, _, _, y_scaler = backpain.load_data()
    return d_train_seq, a_int_1, a_int_2

#Hyperparameter tuning-------------------------------------------------------------------------------------
if __name__ == "__main__":
    stream = open(helpers.get_project_path() + "/hyperparam/config/backpain/config_hyper_deepace_tar.yaml", 'r')
    hyper_config = yaml.safe_load(stream)
    tune_sampler = set_seeds(hyper_config["seed"])
    save_dir_hyper = helpers.get_project_path() + "/hyperparam/parameters/" + hyper_config["dataset"] + "/"
    T = hyper_config["T"]
    #Data
    d_train_seq, a_int_1, a_int_2 = generate_data(hyper_config)
    number_trials = hyper_config["number_trials"]
    methods = hyper_config["methods"]
    for method in methods:
        print(f"Tuning for method {method}")
        if method == "rmsn":
            # RMSN----------
            print("Hyperparameter tuning RMSN Propnet Nominator")
            tune_param_RMSN_nom(d_train_seq, number_trials, save_dir_hyper, tune_sampler=tune_sampler)
            print("Hyperparameter tuning RMSN Propnet Denominator")
            tune_param_RMSN_denom(d_train_seq, number_trials, save_dir_hyper, tune_sampler=tune_sampler)
            # Train optimal propensity networks
            param_prop_nom = joblib.load(save_dir_hyper + "study_rmsn_prop_nom.pkl").best_trial.params
            param_prop_denom = joblib.load(save_dir_hyper + "study_rmsn_prop_denom.pkl").best_trial.params
            prop_nom = rmsn.train_nominator(param_prop_nom, d_train_seq)
            prop_denom = rmsn.train_denominator(param_prop_denom, d_train_seq)
            # Calculate propensity weights
            SW = rmsn.calc_propweights(prop_nom, prop_denom, d_train_seq)
            print("Hyperparameter tuning RMSN Encoder")
            tune_param_RMSN_enc(d_train_seq, SW, number_trials, save_dir_hyper, tune_sampler=tune_sampler)
            # Train optimal encoder
            param_enc = joblib.load(save_dir_hyper + "study_rmsn_enc.pkl").best_trial.params
            rmsn_enc, _ = rmsn.train_rmsn_enc(param_enc, d_train_seq, SW)
            print("Hyperparameter tuning RMSN Decoder")
            tune_param_RMSN_dec(d_train_seq, rmsn_enc, SW, number_trials, save_dir_hyper, tune_sampler=tune_sampler)
        if method == "crn":
            print("Hyperparameter tuning CRN Encoder")
            tune_crn_enc(d_train_seq, number_trials, save_dir_hyper, tune_sampler=tune_sampler)
            # Train optimal encoder network
            param_enc = joblib.load(save_dir_hyper + "study_crn_enc.pkl").best_trial.params
            crn_enc, _ = crn.train_Encoder_CRN(param_enc, d_train_seq)
            print("Hyperparameter tuning CRN Decoder")
            tune_crn_dec(d_train_seq, crn_enc, number_trials, save_dir_hyper, tune_sampler=tune_sampler)
        if method == "gnet":
            print("Hyperparameter tuning GNet")
            tune_gnet(d_train_seq, hyper_config["n_cov_cont"]+1, hyper_config["n_cov_discr"], number_trials, save_dir_hyper,
                      tune_sampler=tune_sampler)
        if method == "lstm_fact":
            print("Hyperparameter tuning Factual LSTM")
            tune_lstm_fact(d_train_seq, number_trials, save_dir_hyper, tune_sampler=tune_sampler)
        if method == "deepace":
            save_dir_hyper = save_dir_hyper + "deepace/"
            study_name = "study_deepace_treat" + str(hyper_config["treat"]) + "_int"
            print("Hyperparameter tuning DeepACE, intervention 1")
            tune_deepace(d_train_seq, a_int_1, study_name=study_name + "1", alpha=0.1, beta=0,
                         num_samples=number_trials, path=save_dir_hyper, tune_sampler=tune_sampler)
            print("Hyperparameter tuning DeepACE, intervention 2")
            tune_deepace(d_train_seq, a_int_2, study_name=study_name + "2", alpha=0.1, beta=0,
                         num_samples=number_trials, path=save_dir_hyper, tune_sampler=tune_sampler)
        if method == "deepace_tar":
            study_name = "study_deepace_tar_treat" + str(hyper_config["treat"]) + "_int"
            save_dir_hyper = save_dir_hyper + "deepace/"
            print("Hyperparameter tuning DeepACE + targeting, intervention 1")
            tune_deepace(d_train_seq, a_int_1, study_name=study_name + "1", alpha=0.1, beta=0.01,
                         num_samples=number_trials, path=save_dir_hyper, tune_sampler=tune_sampler)
            print("Hyperparameter tuning DeepACE + targeting, intervention 2")
            tune_deepace(d_train_seq, a_int_2, study_name=study_name + "2", alpha=0.1, beta=0.01,
                         num_samples=number_trials, path=save_dir_hyper, tune_sampler=tune_sampler)
        if method =="dragonnet":
            print("Hyperparameter tuning DragonNet")
            tune_dragonnet(d_train_seq, number_trials, save_dir_hyper, tune_sampler=tune_sampler, alpha=1, beta=1)
        if method =="tarnet":
            print("Hyperparameter tuning TarNet")
            tune_dragonnet(d_train_seq, number_trials, save_dir_hyper, tune_sampler=tune_sampler, alpha=0, beta=0)