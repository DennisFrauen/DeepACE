import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as fctnl
from sklearn.model_selection import train_test_split
import models.general_models as general
import pytorch_lightning as pl
import math
import models.helpers as helpers


#Objective
def obj_gnet(Lhat_cont, Lhat_discr, x_cont, x_discr):
    loss_cont = torch.mean((Lhat_cont - x_cont)**2)
    if Lhat_discr is not None:
        loss_discr = torch.mean((Lhat_discr - x_discr) ** 2)
    else:
        loss_discr = 0
    return loss_cont + loss_discr


# Training procedure for encoder
def train_gnet(config, data, n_cov_cont, n_cov_discr, epochs=100, tune_mode=False):
    #neptune_logger = NeptuneLogger(project='dennisfrauen/SeqDragonNet')
    # Data
    d_train, d_val = train_test_split(data, test_size=0.2, shuffle=False)
    d_train = torch.from_numpy(d_train.astype(np.float32))
    d_val = torch.from_numpy(d_val.astype(np.float32))
    train_loader = DataLoader(dataset=d_train, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(dataset=d_val, batch_size=config["batch_size"], shuffle=False)
    # Model
    gnet = GNet_model(config=config, input_size=d_train.size(2) - 1, n_cov_cont = n_cov_cont, n_cov_discr = n_cov_discr)
    gnet.set_tune_mode(tune_mode)
    # Train
    Trainer1 = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False)
    Trainer1.fit(gnet, train_loader, val_loader)
    # Validation error after training
    val_results = Trainer1.validate(model=gnet, dataloaders=val_loader, verbose=False)
    val_err = val_results[0]['val_loss']
    return gnet, val_err


#Data input: First discrete covariates, then continuous ones
class GNet_model(general.Causal_Encoder):

    def __init__(self, config, input_size, n_cov_cont, n_cov_discr):
        super().__init__(config, input_size)
        #Head LSTMs
        self.demux_size_cont = math.ceil(config["hidden_size_lstm"]* (n_cov_cont /( n_cov_cont + n_cov_discr)))
        self.demux_size_diskr = config["hidden_size_lstm"] - self.demux_size_cont
        self.n_cov_cont = n_cov_cont
        self.n_cov_discr = n_cov_discr
        self.lstm_cont = general.VariationalLSTM(input_size=self.demux_size_cont, hidden_size=config["hidden_size_lstm2"], num_layers=1,
                                    dropout_rate=config["dropout"])
        self.linear_cont = nn.Linear(in_features=config["hidden_size_lstm2"], out_features=n_cov_cont)
        self.lstm_diskr = general.VariationalLSTM(input_size=self.demux_size_diskr + n_cov_cont, hidden_size=config["hidden_size_lstm2"], num_layers=1,
                                    dropout_rate=config["dropout"])
        self.linear_diskr = nn.Linear(in_features=config["hidden_size_lstm2"], out_features=n_cov_discr)
        # Optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])

    def format_input(self, data_torch):
        T = data_torch.size(1)
        enc_input = data_torch[:, :T-1, 1:]
        x_diskr = data_torch[:, 1:, 2:(2+self.n_cov_discr)]
        x_cont = data_torch[:, 1:, (2+self.n_cov_discr):]
        return enc_input, x_cont, x_diskr

    def forward(self, input_fwd, teacher_forcing = False):
        x = input_fwd[0]
        x_cont = input_fwd[1]

        #Initialize hidden states
        h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size)

        # Forward propagate RNN
        R, _ = self.lstm(x, (h0, c0))
        # R: tensor of shape (batch_size, seq_length, hidden_size) - Outputs are hidden states

        #Selector/ Demux
        R_cont = R[:, :, 0:self.demux_size_cont]
        R_diskr = R[:, :, self.demux_size_cont:]

        #Head LSTMs
        #Continuous
        h0_cont = torch.zeros(1, R_cont.size(0), self.lstm_cont.hidden_size)
        c0_cont = torch.zeros(1, R_cont.size(0), self.lstm_cont.hidden_size)
        out_cont, _ = self.lstm_cont(R_cont, (h0_cont, c0_cont))
        Lhat_cont = self.linear_cont(out_cont)
        if self.n_cov_discr > 0:
            #Discrete
            h0_diskr = torch.zeros(1, R_diskr.size(0), self.lstm_diskr.hidden_size)
            c0_diskr = torch.zeros(1, R_diskr.size(0), self.lstm_diskr.hidden_size)
            input_diskr = torch.zeros(size=(x.size(0), x.size(1), Lhat_cont.size(2) + R_diskr.size(2)))
            input_diskr[:, :, 0:R_diskr.size(2)] = R_diskr
            if teacher_forcing:
                input_diskr[:, :, R_diskr.size(2):] = x_cont
            else:
                input_diskr[:, :, R_diskr.size(2):] = Lhat_cont
            out_discr, _ = self.lstm_diskr(input_diskr, (h0_diskr, c0_diskr))
            Lhat_discr = self.linear_cont(out_discr)
        else:
            Lhat_discr = None
        return Lhat_cont, Lhat_discr

    def training_step(self, train_batch, batch_idx):
        self.train()
        batch_size = train_batch.size(0)
        input_fwd = self.format_input(train_batch)
        x_cont = input_fwd[1]
        x_discr = input_fwd[2]
        # Forward pass
        Lhat_cont, Lhat_discr = self.forward(input_fwd, teacher_forcing=True)
        # Loss
        loss = obj_gnet(Lhat_cont, Lhat_discr, x_cont, x_discr)
        if not self.tune_mode:
            self.log('train_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        batch_size = val_batch.size(0)
        input_fwd = self.format_input(val_batch)
        x_cont = input_fwd[1]
        x_discr = input_fwd[2]
        # Forward pass
        Lhat_cont, Lhat_discr = self.forward(input_fwd, teacher_forcing=True)
        # Loss
        loss = obj_gnet(Lhat_cont, Lhat_discr, x_cont, x_discr)
        self.log('val_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def estimate_ace(self, d_train_seq, d_holdout, a_int_1, a_int_2, K = 1000, y_scaler=None):
        #a_int_1 = np.ones(d_train_seq.shape[0])
        #a_int_2 = np.zeros(d_train_seq.shape[0])
        self.eval()
        d_train_torch = torch.from_numpy(d_train_seq.astype(np.float32))
        d_holdout_torch = torch.from_numpy(d_holdout.astype(np.float32))
        a_int_1_torch = torch.from_numpy(a_int_1.astype(np.float32))
        a_int_2_torch = torch.from_numpy(a_int_2.astype(np.float32))
        with torch.no_grad():
            y_hat_1 = self.monte_carlo(d_train_torch[:, :, 2:], d_holdout_torch[:, :, 2:], a_int_1_torch, K)
            y_hat_2 = self.monte_carlo(d_train_torch[:, :, 2:], d_holdout_torch[:, :, 2:], a_int_2_torch, K)

        #Rescale
        y_hat_1_unscaled = helpers.rescale_y(y_hat_1, y_scaler)
        y_hat_2_unscaled = helpers.rescale_y(y_hat_2, y_scaler)
        #Estimate
        diff = y_hat_1_unscaled - y_hat_2_unscaled
        ace = np.mean(diff)
        return ace

    def monte_carlo(self, x_train, x_holdout_cont, a_int, K):
        T = x_train.size(1)
        n = x_train.size(0)
        n_holdout = x_holdout_cont.size(0)
        y_hat_mc = torch.zeros(n, K)
        #L_hat_mc = torch.zeros(n, T, self.n_cov_cont + self.n_cov_discr)
        L_hat_t = x_train[:, 0:1, :]
        A_int = torch.tile(torch.unsqueeze(a_int, 0), (n, 1))
        for k in range(K):
            #Initialize hidden states
            h = torch.zeros(1, n, self.lstm.hidden_size)
            c = torch.zeros(1, n, self.lstm.hidden_size)
            h_cont = torch.zeros(1, n, self.lstm_cont.hidden_size)
            c_cont = torch.zeros(1, n, self.lstm_cont.hidden_size)
            for t in range(T):
                # Forward pass
                L_hat_t, h, c, h_cont, c_cont = self.forward_mc(L_hat_t, A_int[:, t], h, c, h_cont, c_cont)
                for i in range (n):
                    if t < T -1:
                        # Residue distribution for patient i
                        res_ti = x_holdout_cont[:, t+1, :] - torch.tile(L_hat_t[i:(i+1), 0, :], (n_holdout, 1))
                        #Sample from residue distribution
                        j = random.randint(0, n_holdout - 1)
                        L_hat_t[i, 0, :] = L_hat_t[i, 0, :] + res_ti[j, :]
                    else:
                        #Last time step, save monto carlo estimates for last outcome
                        y_hat_mc[i, k] = L_hat_t[i, 0, -1]
        y_hat = np.mean(y_hat_mc.detach().numpy(), 1)
        return y_hat

    #Forward pass over a single timestep (when doing monte carlo sampling)
    def forward_mc(self, L_hat_t, A_int_t, h_t, c_t, h_cont_t, c_cont_t):
        A_int_t = torch.unsqueeze(torch.unsqueeze(A_int_t, 1), 2)
        input_lstm = torch.concat((A_int_t, L_hat_t), 2)
        R, (h_out, c_out) = self.lstm(input_lstm, (h_t, c_t))

        #Second lstm for continous cov
        out_cont, (h_out_cont, c_out_cont) = self.lstm_cont(R, (h_cont_t, c_cont_t))
        Lhat_cont_out = self.linear_cont(out_cont)

        return Lhat_cont_out, h_out, c_out, h_out_cont, c_out_cont