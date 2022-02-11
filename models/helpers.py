import numpy as np
import torch
import sklearn as sk
import optuna
import joblib
from pathlib import Path
import os

def get_project_path():
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    return str(path.parent.absolute())
#Standardizes data and returns skalar object (for rescaling when predicting)
def standardize_data(data_seq):
    #input size (n,T,p+2)
    n = data_seq.shape[0]
    T = data_seq.shape[1]
    #Remove treatments (not scaled)
    y_x = np.delete(data_seq,1,2)
    y_scaler = None
    #Go through outcomes + covariates (except last column which contains lagged outcomes)
    for i in range(y_x.shape[2] - 1):
        vec_unscaled = np.reshape(y_x.copy()[:,:,i],(n*T,1))
        scaler = sk.preprocessing.StandardScaler().fit(vec_unscaled)
        #Save outcomes scaler
        if i == 0:
            y_scaler = scaler
        vec_scaled = scaler.transform(vec_unscaled)[:, 0:1]
        y_x[:,:,i] = np.reshape(vec_scaled.copy(),(n,T))
    #Update the lagged outcomes column with scaled lagged outcomes
    y_x[:,0,-1] = np.zeros(n)
    y_x[:,1:,-1] = y_x[:,0:(T-1),0]
    #Add treatments
    data_seq[:,:,0] = y_x[:,:,0]
    data_seq[:,:,2:] = y_x[:,:,1:]
    return data_seq, y_scaler

def standardize_covariates(data_seq):
    #input size (n,T,p)
    n = data_seq.shape[0]
    T = data_seq.shape[1]
    p = data_seq.shape[2]
    data_scaled = data_seq.copy()
    for i in range(p):
        vec_unscaled = np.reshape(data_scaled[:,:,i],(n*T,1))
        scaler = sk.preprocessing.StandardScaler().fit(vec_unscaled)
        vec_scaled = scaler.transform(vec_unscaled)[:, 0:1]
        data_scaled[:,:,i] = np.reshape(vec_scaled.copy(),(n,T))
    return data_scaled

#Input (n, T)
def standardize_outcome(data_seq):
    n = data_seq.shape[0]
    T = data_seq.shape[1]
    vec_unscaled = np.reshape(data_seq.copy(), (n * T, 1))
    scaler = sk.preprocessing.StandardScaler().fit(vec_unscaled)
    vec_scaled = scaler.transform(vec_unscaled)[:, 0:1]
    data_scaled = np.reshape(vec_scaled.copy(), (n, T))
    return data_scaled, scaler

#Rescales outcomes
def rescale_y(y_data_scaled, y_scaler):
    if y_scaler is not None:
        n = y_data_scaled.shape[0]
        T = 1
        if y_data_scaled.ndim > 1:
            T = y_data_scaled.shape[1]
        else:
            y_data_scaled = np.expand_dims(y_data_scaled,1)
        y_vec_scaled = np.reshape(y_data_scaled.copy(), (n * T, 1))
        y_vec_unscaled = y_scaler.inverse_transform(y_vec_scaled)
        y_data_unscaled = np.reshape(y_vec_unscaled, (n, T))

        if T == 1:
            y_data_unscaled = np.squeeze(y_data_unscaled)
        return y_data_unscaled
    else:
        return y_data_scaled


#Returns true ATEs from counterfactual data
def estimate_ates_from_cf(data_cf1,data_cf2,tau):
    T = data_cf1.shape[1]
    y_int1 = data_cf1[:,T-tau:,0]
    y_int2 = data_cf2[:, T - tau:, 0]
    return np.mean(y_int1 - y_int2, axis=0)


#Returns estimated ACE trajectory given a trained encoder-decoder model-------------------------------
#Needed: encoder.predict_1step(data,a_int,y_scaler), returns g_hat, y_hat, repr
#        decoder.predict(a_int, repr_enc, y_hat_enc), returns g_hat, y_hat
def estimate_ates(d_train_seq, p_static, a_int1, a_int2, encoder, decoder = None, tau_enc=1, y_scaler=None):
    n = d_train_seq.shape[0]
    T = d_train_seq.shape[1]
    tau = a_int1.shape[0]

    #Estimate encoder ATEs (interventions in observational history and 1-step ahead ATE)
    ates_enc = np.empty(tau_enc)
    #Start at time T-tau_enc, then estimate 1-step ahead ATE for every timestep
    data1 = d_train_seq
    data2 = d_train_seq
    for step in range(tau_enc):
        # Present intervention
        a_t1 = a_int1[step:step + 1]
        a_t2 = a_int2[step:step + 1]
        if step > 0:
            #Further prediction steps: Keep trajectories where observed treatment history coincides with intervention
            data_1_ind = []
            data_2_ind = []
            for i in range(n):
                a_observed = d_train_seq[i, T - tau_enc:T - tau_enc + step, 1]
                # Check whether observed treatment history coincides with intervention
                if np.array_equal(a_observed, a_int1[:step]):
                    data_1_ind.append(True)
                else:
                    data_1_ind.append(False)
                if np.array_equal(a_observed, a_int2[:step]):
                    data_2_ind.append(True)
                else:
                    data_2_ind.append(False)
            data1 = d_train_seq[data_1_ind, :, :]
            data2 = d_train_seq[data_2_ind, :, :]
        if data1.shape[0] > 0 and data2.shape[0] > 0:
            pred1 = encoder.predict_1step(data=data1[:, :T - tau_enc + step + 1, :], a_int=a_t1, y_scaler=y_scaler)
            pred2 = encoder.predict_1step(data=data2[:, :T - tau_enc + step + 1, :], a_int=a_t2, y_scaler=y_scaler)
            y_hat1, y_hat2 = pred1[0], pred2[0]
            ates_enc[step] = np.mean(y_hat1[:, -1]) - np.mean(y_hat2[:, -1])
        else:
            ates_enc[step] = np.nan

    #Estimate decoder ATE's (if trained decoder provided)
    if decoder is None:
        return ates_enc
    else:
        #Check whether there is a pre decoder intervention, if not use observational treatments as first decoder input
        if tau_enc > 0:
            a_int1_enc = a_int1[tau_enc - 1]
            a_int2_enc = a_int2[tau_enc - 1]
            A_int1 = np.tile(a_int1[tau_enc - 1:], (data1.shape[0], 1))
            A_int2 = np.tile(a_int2[tau_enc - 1:], (data2.shape[0], 1))
        else:
            a_int1_enc = None
            a_int2_enc = None
            A_int1 = np.concatenate((data1[:,-1,1:2], np.tile(a_int1, (data1.shape[0], 1))),axis=1)
            A_int2 = np.concatenate((data2[:,-1,1:2], np.tile(a_int2, (data2.shape[0], 1))),axis=1)
        # build representation and 1-step ahead forecast with encoder
        pred1 = encoder.predict_1step(data=data1, a_int=a_int1_enc)
        pred2 = encoder.predict_1step(data=data2, a_int=a_int2_enc)
        y_hat_enc1, y_hat_enc2 = pred1[0], pred2[0]
        repr1, repr2 = pred1[1], pred2[1]
        #Static covariates
        p = d_train_seq.shape[2] - 3 - p_static
        X_static1 = data1[:,0,p+1:p+1+p_static]
        X_static2 = data2[:, 0, p + 1:p + 1 + p_static]
        #Predict using decoder
        _, y_hat_dec1 = decoder.predict(A_int=A_int1, repr_enc=repr1, y_hat_enc=y_hat_enc1[:, -1],X_static = X_static1, y_scaler=y_scaler)
        _, y_hat_dec2 = decoder.predict(A_int=A_int2, repr_enc=repr2, y_hat_enc=y_hat_enc2[:, -1],X_static = X_static2, y_scaler=y_scaler)
        # Estimated ATE
        ates_dec = np.mean(y_hat_dec1, axis=0) - np.mean(y_hat_dec2, axis=0)
        return np.concatenate((ates_enc, ates_dec))


#Explodes dataset into time batches, used for decoder training
def explode_dataset(data, encoder, tau_max, batch_size):
    n = data.shape[0]
    T = data.shape[1]
    np.random.shuffle(data)
    batches = np.array_split(data, int(n / batch_size))
    # Expode dataset
    time_batches = []
    for batch in batches:
        # Split each batch into minibatches of size (batchsize, tau_max) through time
        for t in range(2, T - tau_max + 1):

            # Compute representation and one-step-ahead prediction using encoder
            _, y_hat, [h_t, c_t] = encoder.predict_1step(batch[:, :t, :])
            enc_output_t = np.empty((batch_size, tau_max+1, 2 * encoder.lstm.hidden_size + 1))
            enc_output_t[:, 0, :(2 * encoder.lstm.hidden_size)] = np.squeeze(np.concatenate((h_t, c_t), axis=2))
            enc_output_t[:, 0, -1] = y_hat[:, -1]
            # Data for decoder, batches have length tau_max+1 (starting at previous timestep)
            batch_t = batch[:, (t-1):(t + tau_max), :]
            # Save history and decoder input
            time_batches.append(np.concatenate((batch_t, enc_output_t), axis=2))
    np.random.shuffle(time_batches)
    data_expl = np.concatenate(time_batches, axis=0)
    # data expl is of shape (n*(T-tau),tau,p+2+ dim(h_t) + dim(c_t)
    # representations and encoder one-step ahead prediction for each batch/ timestep are saved in data_exp[:,0,(p+2):]
    # Train/ test split (80% training data)
    n_exp = data_expl.shape[0]
    batch_nr = int(n_exp / batch_size)
    batch_nr_train = (0.8 * batch_nr)
    d_train = data_expl[0:int(batch_nr_train * batch_size), :, :]
    d_val = data_expl[int(batch_nr_train * batch_size):, :, :]
    return torch.from_numpy(d_train.astype(np.float32)), torch.from_numpy(d_val.astype(np.float32))

#Extracts data from exploded batch, used for decoder training
#Input: Batch Tensor of expoded data
def format_exploded_batch(train_batch, hidden_size_lstm_enc, p_static):
    #Dimensions
    T = train_batch.size(1)
    hid = hidden_size_lstm_enc
    p = train_batch.size(2) - 2 * hid - 3 - p_static
    p_total = p + p_static
    # Input for decoder
    a_input = train_batch[:, 0:(T - 1),1:2]
    y_data_lag = train_batch[:, 0:(T - 1), 0:1]
    X_static = train_batch[:, 0:(T-1), (p + 1):(p_total + 1)]
    dec_input = torch.cat((a_input, X_static, y_data_lag), 2)
    #Labels (for loss),i.e. shifted treatments and outcomes
    y_data = train_batch[:, 1:, 0]
    a_data = train_batch[:, 1:, 1]
    # Representations from encoder
    h_t = torch.unsqueeze(train_batch[:, 0, (p_total + 2):(p_total + 2 + hid)], 0)
    c_t = torch.unsqueeze(train_batch[:, 0, (p_total + 2 + hid):(p_total + 2 + 2 * hid)], 0)
    repr_enc = [h_t, c_t]
    # One-step-ahead prediction from encoder
    y_hat_enc = train_batch[:, 0, -1]

    return dec_input, y_data, a_data, repr_enc, y_hat_enc


#Input: Objective function depending on trial
def tune_hyperparam(objective, study_name, path, num_samples=10, sampler=None):
    if sampler is not None:
        study = optuna.create_study(direction="minimize", study_name=study_name, sampler=sampler)
    else:
        study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(objective, n_trials=num_samples)
    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial_best = study.best_trial

    print("  Value: ", trial_best.value)

    print("  Params: ")
    for key, value in trial_best.params.items():
        print("    {}: {}".format(key, value))

    save_dir = path + study_name + ".pkl"
    joblib.dump(study, save_dir)

