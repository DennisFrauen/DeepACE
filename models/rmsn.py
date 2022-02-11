# %% Recurrent Marginal Structural Networks
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as fctnl
from sklearn.model_selection import train_test_split
import optuna
import joblib
from pathlib import Path
import os
import models.general_models as general
import pytorch_lightning as pl
import models.helpers as helpers

# Helful functions---------------------------------------------------------------------------

# Treatment possibilites: in case k = 1, only 2 possibilities
def all_comb(length):
    return np.array(np.meshgrid(*[[0, 1]] * length, indexing='ij')).reshape((length, -1)).transpose()


# Data preprocessing for propensity network training
def get_data_pw(Y_train, A_train, X_train):
    n, T, k = A_train.shape
    p = X_train.shape[2]
    a_comb = all_comb(k)
    # Data for estimating propensity weights (treatment classification)
    data_train = np.concatenate([Y_train, A_train, X_train], axis=2)
    data_pw = data_train
    a_class = np.empty(shape=(n, T, 1))
    data_pw = np.concatenate([data_pw, a_class], axis=2)

    # Transform to multi-class classification problem
    for pat in range(n):
        for t in range(T):
            for v in range(2 ** k):
                if np.array_equal(data_pw[pat, t, 1:(1 + k)], a_comb[v]):
                    data_pw[pat, t, 1 + k + p] = v

    data_pw = torch.from_numpy(data_pw.astype(np.float32))
    return data_pw


# Models------------------------------------------------------------------------------------
# Propensity network
# Description: estimating stabilized propensity weights by predicting treatment using LSTM architecture
class PropNet(nn.Module):
    def __init__(self, input_size, num_classes, params):  # hidden_size, num_classes, num_layers):
        super(PropNet, self).__init__()
        self.num_layers = 1
        #self.lstm = nn.LSTM(input_size, params["hidden_size_lstm"], 1, batch_first=True)
        self.lstm = general.VariationalLSTM(input_size=input_size, hidden_size=params["hidden_size_lstm"], num_layers=1,
                                    dropout_rate=params["dropout"])
        # -> x needs to be: (batch_size, seq_length, input_size)
        self.fc = nn.Linear(params["hidden_size_lstm"], num_classes)

    def forward(self, x):
        # Device configuration
        h0 = torch.zeros(self.num_layers, x.size(0), self.lstm.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.lstm.hidden_size)
        # x: (batch_size, seq_len, input_size), h0: (num_layers, batch_size, hidden_size1)

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size1) - Outputs are hidden states

        out = torch.sigmoid(self.fc(out))
        # out: tensor of shape (batch_size, seq_length, num_classes)

        return out


# Propensity network nominator-------------------------------------------------------------------------
class PropNetNom(PropNet):
    def __init__(self, params):
        super().__init__(1, 1, params)

    def validation_step(self, d_val):
        self.eval()
        T = d_val.shape[1]
        a_input = torch.from_numpy(d_val[:, 0:(T - 1), 1:2].astype(np.float32))
        out = torch.squeeze(self.forward(a_input))
        # out = torch.transpose(out, 1, 2)
        target = get_data_pw(d_val[:, :, 0:1], d_val[:, :, 1:2], d_val[:, :, 2:])[:, 1:T, -1]
        target = target.float()
        loss = nn.BCELoss()
        err = loss(out, target)
        return err.detach().numpy()


# Train propensity network for nominator and binary treatment
def train_nominator(params, data, epochs=100):
    Y_train = data[:, :, 0:1]
    A_train = data[:, :, 1:2]
    X_train = data[:, :, 2:]
    p = X_train.shape[2]
    T = X_train.shape[1]
    k = 1

    # Trainsform to multiclass classification problem
    data_pw = get_data_pw(Y_train, A_train, X_train)
    train_loader_pw = DataLoader(dataset=data_pw, batch_size=params["batch_size"], shuffle=True)

    prop_net_nom = PropNetNom(params)
    prop_net_nom.train()
    optimizer = torch.optim.Adam(prop_net_nom.parameters(), lr=params["lr"])

    # Train
    print('Train Nominator network')
    for epoch in range(epochs):
        for batch in train_loader_pw:
            y_data = batch[:, 1:T, (1 + k + p)]
            y_data = y_data.float()
            x_data = batch[:, 0:(T - 1), 1:(1 + k)]
            # Forward pass
            out = torch.squeeze(prop_net_nom(x_data))
            # out = torch.transpose(out, 1, 2)
            loss = nn.BCELoss()
            loss = loss(out, y_data)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return prop_net_nom

# Propensity network denominator----------------------------------------------------------------------------------
class PropNetDeNom(PropNet):
    def __init__(self, params, input_size):
        super().__init__(input_size, 1, params)

    def validation_step(self, d_val):
        self.eval()
        T = d_val.shape[1]
        input = np.concatenate((d_val[:, 0:(T - 1), 1:(2)], d_val[:, 1:T, 2:]), axis=2)
        input = torch.from_numpy(input.astype(np.float32))
        out = torch.squeeze(self.forward(input))
        # out = torch.transpose(out, 1, 2)
        target = get_data_pw(d_val[:, :, 0:1], d_val[:, :, 1:2], d_val[:, :, 2:])[:, 1:T, -1]
        target = target.float()
        loss = nn.BCELoss()
        err = loss(out, target)
        return err.detach().numpy()


# Train propensity nework for denominator
def train_denominator(params, data, epochs=100):
    Y_train = data[:, :, 0:1]
    A_train = data[:, :, 1:2]
    X_train = data[:, :, 2:]
    T = X_train.shape[1]
    p = X_train.shape[2]
    k = 1

    # Trainsform to multiclass classification problem
    data_pw = get_data_pw(Y_train, A_train, X_train)
    train_loader_pw = DataLoader(dataset=data_pw, batch_size=params["batch_size"], shuffle=True)

    # Model
    prop_net_denom = PropNetDeNom(params, input_size=p + 1)
    prop_net_denom.train()
    # Estimate denominator
    obj = nn.BCELoss()
    optimizer = torch.optim.Adam(prop_net_denom.parameters(), lr=params["lr"])

    # Train
    print('Train Denominator network')
    for epoch in range(epochs):
        for batch in train_loader_pw:
            y_data = batch[:, 1:T, (1 + k + p)]
            y_data = y_data.float()
            x_data = torch.cat((batch[:, 0:(T - 1), 1:(1 + k)],
                                batch[:, 1:T, (1 + k):(1 + k + p)]), dim=2)
            # Forward pass
            out = torch.squeeze(prop_net_denom(x_data))
            # out = torch.transpose(out, 1, 2)
            loss = obj(out, y_data)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return prop_net_denom


# Encoder RMSN---------------------------------------------------------------------------------------------------
class Encoder(general.Causal_Encoder):

    def __init__(self, config, input_size, SW):
        super().__init__(config, input_size)
        # -> x needs to be: (batch_size, seq_length, input_size)
        #self.fc1 = nn.Linear(config["hidden_size_lstm"], config["hidden_size"])
        self.fc2 = nn.Linear(config["hidden_size_lstm"], 1)
        # Optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        #Weights
        self.SW = torch.from_numpy(SW.astype(np.float32))

    def format_input(self, data_torch):
        enc_input = data_torch[:,:,1:]
        y_data = data_torch[:,:,0]
        a_data = data_torch[:,:,1]
        return enc_input, y_data, a_data

    def forward(self, input_fwd):
        x = input_fwd[0]
        h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size)
        # x: (batch_size, seq_len, input_size), h0: (num_layers, batch_size, hidden_size1)

        # Forward propagate RNN
        out, (h_t, c_t) = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size) - Outputs are hidden states
        #y_hidden = fctnl.elu(self.fc1(out))
        y_out = self.fc2(out)[:, :, 0]
        # y_out: tensor of shape (batch_size,seq_length)

        # h_t and c_t output are of size (batch_size,hidden size)
        return y_out, [h_t, c_t]

    def training_step(self, train_batch, batch_idx):
        self.train()
        batch_size = train_batch.size(0)
        input_fwd = self.format_input(train_batch)
        y_data = input_fwd[1]
        # Forward pass
        y_hat, _ = self.forward(input_fwd)
        # Loss
        loss = torch.mean(self.SW[(batch_idx * batch_size):((batch_idx + 1) * batch_size), :, 0] * (y_hat - y_data) ** 2)
        self.log('train_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        batch_size = val_batch.size(0)
        input_fwd = self.format_input(val_batch)
        y_data = input_fwd[1]
        # Forward pass
        y_hat, _ = self.forward(input_fwd)
        # Loss
        loss = torch.mean((y_hat - y_data) ** 2)
        self.log('val_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss


# Encoder training
def train_rmsn_enc(config, data, SW, epochs=100, tune_mode=False):
    # Data
    d_train, d_val = train_test_split(data, test_size=0.2, shuffle=False)
    SW_train, SW_val = train_test_split(SW, test_size=0.2, shuffle=False)
    d_train = torch.from_numpy(d_train.astype(np.float32))
    d_val = torch.from_numpy(d_val.astype(np.float32))
    train_loader = DataLoader(dataset=d_train, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(dataset=d_val, batch_size=config["batch_size"], shuffle=False)
    # Model
    enc_rmsn = Encoder(config=config, input_size=d_train.size(2) - 1, SW=SW_train)
    enc_rmsn.set_tune_mode(tune_mode)

    # Train
    Trainer1 = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False)
    Trainer1.fit(enc_rmsn, train_loader, val_loader)
    # Validation error after training
    val_results = Trainer1.validate(model=enc_rmsn, dataloaders=val_loader, verbose=False)
    val_err = val_results[0]['val_loss']

    return enc_rmsn, val_err

# Decoder RMSN-------------------------------------------------------------------------------------------------------
class Decoder_rmsn(general.Causal_Decoder):
    def __init__(self, config, encoder, SW):
        super().__init__(config, encoder, input_size=1)
        self.fc = nn.Linear(config["hidden_size_lstm"], 1)
        # Optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        #Weights
        self.SW = torch.from_numpy(SW.astype(np.float32))
        self.encoder = encoder

    def forward(self, input_fwd, tf=True):
        repr_enc = input_fwd[0]
        A = input_fwd[2]
        # Memory adapter -> hidden state initialization from encoder
        h = fctnl.elu(self.adapter1(repr_enc[0]))
        c = fctnl.elu(self.adapter2(repr_enc[1]))

        #h, c: (num_layers=1, batch_size, hidden_size)

        lstm_input = torch.unsqueeze(A[:, 1:], 2)
        lstm_out, _ = self.lstm(lstm_input, (h, c))

        y_hat = torch.squeeze(self.fc(lstm_out))

        return [y_hat]

    # Model Training
    def training_step(self, train_batch, batch_idx):
        batch_size = train_batch.size(0)
        input_fwd = self.format_input(train_batch)
        y_data = input_fwd[2]
        # Forward pass
        y_hat = self.forward(input_fwd)
        #Loss
        #Take sw at time t=1 for tau \in {1, ..., T}
        loss = torch.mean(self.SW[(batch_idx * batch_size):((batch_idx + 1) * batch_size), 1, 1:] * (y_hat[0] - y_data[:, 1:]) ** 2)
        if not self.tune_mode:
            # Logging
            self.log('train_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    # Model validation
    def validation_step(self, train_batch, batch_idx):
        input_fwd = self.format_input(train_batch)
        y_data = input_fwd[2]
        # Forward pass
        y_hat = self.forward(input_fwd)
        # Loss
        loss = torch.mean((y_hat[0] - y_data[:, 1:])**2)
        # Logging
        self.log('val_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

# Encoder training
def train_rmsn_dec(config, data, encoder, SW, epochs=100, tune_mode=False):
    # Data
    d_train, d_val = train_test_split(data, test_size=0.2, shuffle=False)
    SW_train, SW_val = train_test_split(SW, test_size=0.2, shuffle=False)
    d_train = torch.from_numpy(d_train.astype(np.float32))
    d_val = torch.from_numpy(d_val.astype(np.float32))
    train_loader = DataLoader(dataset=d_train, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(dataset=d_val, batch_size=config["batch_size"], shuffle=False)
    # Model
    dec_rmsn = Decoder_rmsn(config=config, encoder=encoder, SW=SW_train)
    dec_rmsn.set_tune_mode(tune_mode)

    # Train
    Trainer1 = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False)
    Trainer1.fit(dec_rmsn, train_loader, val_loader)
    # Validation error after training
    val_results = Trainer1.validate(model=dec_rmsn, dataloaders=val_loader, verbose=False)
    val_err = val_results[0]['val_loss']

    return dec_rmsn, val_err

#Joint training with propensity networks
def train_rmsn_joint(params_nom, params_denom, params_enc, params_dec, data, epochs=100):
    #Train optimal propensity networks
    prop_nom = train_nominator(params_nom,data, epochs)
    prop_denom = train_denominator(params_denom, data, epochs)
    SW = calc_propweights(prop_nom, prop_denom, data)
    encoder, _ = train_rmsn_enc(params_enc, data, SW, epochs)
    decoder, _ = train_rmsn_dec(params_dec, data, encoder, SW, epochs=epochs)
    return encoder, decoder

# Calculate propensity weights using trained nominator and denominator propoensity networks
def calc_propweights(mod_PN_num, mod_PN_den, data):
    # Initialization
    Y_train = data[:, :, 0:1]
    A_train = data[:, :, 1:2]
    X_train = data[:, :, 2:]
    n = X_train.shape[0]
    T = X_train.shape[1]
    p = X_train.shape[2]
    k = 1

    data_pw = get_data_pw(Y_train, A_train, X_train)

    # Numerator
    fm_At = mod_PN_num(data_pw[:, 0:(T - 1), 1:(1 + k)])
    fm_At = fm_At.cpu().detach().numpy()
    # Denominator
    fm_AtHt = mod_PN_den(torch.cat((data_pw[:, 0:(T - 1), 1:(1 + k)],
                                    data_pw[:, 1:T, (1 + k):(1 + k + p)]), dim=2))
    fm_AtHt = fm_AtHt.cpu().detach().numpy()

    # fm_At amd fm_AtHt are of size (n,T-1,2^k) <-> P(A_t = 1 |...) and P(A_t = 0 |...) for t = 1,...,T and all n  if k = 1
    tau_max = T
    SW_rmsn = np.zeros((n, T, tau_max))
    SW_rmsn[:, 0, 0] = torch.ones(n)
    fn_frac = fm_At / fm_AtHt
    for tau in range(tau_max):
        for t in range(T - tau):
            if t>0 or tau>0:
                SW_rmsn[:, t, tau] = np.prod(fn_frac[:, t:(t+tau+1), 0], axis=1)
                # Truncating weights at percentiles
                perc_1 = np.percentile(SW_rmsn[:, t, tau], q=1)
                perc_99 = np.percentile(SW_rmsn[:, t, tau], q=99)
                for i in range(n):
                    if SW_rmsn[i, t, tau] < perc_1:
                        SW_rmsn[i, t, tau] = perc_1
                    elif SW_rmsn[i, t, tau] > perc_99:
                        SW_rmsn[i, t, tau] = perc_99
    #Normalising, dividing by mean for each fixed prediction horizon
    #for tau in range(tau_max):
    #    SW_rmsn[:, :(T-tau), tau] = SW_rmsn[:, :(T-tau), tau] / np.mean(SW_rmsn[:, :(T-tau), tau])
    return SW_rmsn


