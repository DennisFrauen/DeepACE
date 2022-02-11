import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as fctnl
from sklearn.model_selection import train_test_split
import models.general_models as general
import pytorch_lightning as pl


# Gradient reversal-----------------------------------------------------
class RevGrad_fcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, lamb):
        ctx.save_for_backward(input_, lamb)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, lamb = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * lamb
        return grad_input, None


class RevGrad(torch.nn.Module):
    def __init__(self):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__()

    def forward(self, input_, lamb):
        revgrad = RevGrad_fcn.apply
        return revgrad(input_, lamb)


# CRN objective
def obj_crn(a_hat, a_data, y_hat, y_data, alpha=1):
    outcome_loss = torch.mean((y_hat - y_data) ** 2)
    treat_loss = fctnl.binary_cross_entropy(a_hat, a_data, reduction='mean')
    return outcome_loss + alpha * treat_loss, outcome_loss, treat_loss


# Encoder------------------------------------------------------------
# Training procedure for encoder
def train_Encoder_CRN(config, data, epochs=100, alpha=1, tune_mode=False):
    # Data
    d_train, d_val = train_test_split(data, test_size=0.2, shuffle=False)
    d_train = torch.from_numpy(d_train.astype(np.float32))
    d_val = torch.from_numpy(d_val.astype(np.float32))
    train_loader = DataLoader(dataset=d_train, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(dataset=d_val, batch_size=config["batch_size"], shuffle=False)
    # Model
    enc_crn = Encoder_crn(config=config, input_size=d_train.size(2) - 1, alpha=alpha)
    enc_crn.set_tune_mode(tune_mode)
    # Train
    Trainer1 = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False)
    Trainer1.fit(enc_crn, train_loader, val_loader)
    # Validation error after training
    val_results = Trainer1.validate(model=enc_crn, dataloaders=val_loader, verbose=False)
    val_err = val_results[0]['val_loss']
    return enc_crn, val_err


# Encoder model
class Encoder_crn(general.Causal_Encoder):
    def __init__(self, config, input_size, alpha=1):
        super().__init__(config, input_size)
        self.body1 = nn.Linear(config["hidden_size_lstm"], config["hidden_size_body"])
        # Outcome predictor
        self.headylayer1 = nn.Linear(config["hidden_size_body"] + 1, config["hidden_size_head"])
        self.headylayer2 = nn.Linear(config["hidden_size_head"], config["hidden_size_head"])
        self.headylayer3 = nn.Linear(config["hidden_size_head"], 1)
        self.reversal = RevGrad()
        self.headalayer1 = nn.Linear(config["hidden_size_body"], config["hidden_size_head"])
        self.headalayer2 = nn.Linear(config["hidden_size_head"], config["hidden_size_head"])
        self.headalayer3 = nn.Linear(config["hidden_size_head"], 1)
        # Optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        # Indicator whether representation balancing is applied
        self.alpha = alpha

    def format_input(self, data_torch):
        T = data_torch.size(1)
        # train_batch is of size (batch_size, seq_len,input_size)
        y_data = data_torch[:, :, 0]
        a_data = data_torch[:, :, 1]
        x_data = data_torch[:, :, 2:]
        a_input = torch.empty((data_torch.size(0), T, 1))
        a_input[:, 0, 0] = torch.zeros(data_torch.size(0))
        a_input[:, 1:, 0] = a_data[:, 0:(T - 1)]
        enc_input = torch.cat((a_input, x_data), 2)
        return enc_input, y_data, a_data

    def forward(self, input_fwd):
        x = input_fwd[0]
        a_data = input_fwd[2]
        # x is of size (batch_size, seq_len,input_size), treat_ind is of size (batch_size, seq_len)
        h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size)
        # lstm_out: tensor of shape (batch_size, seq_length, hidden_size_lstm) - Outputs are hidden states
        lstm_out, (h_t, c_t) = self.lstm(x, (h0, c0))
        # Dragon body
        repr = fctnl.elu(self.body1(lstm_out))

        # Add current treatment as input for outcome predictor network
        repr_y = torch.concat((repr, torch.unsqueeze(a_data, 2)), 2)
        # Outcome prediction network
        y_hat = fctnl.elu(self.headylayer1(repr_y))
        y_hat = torch.squeeze(self.headylayer3(y_hat))
        # Gradient reversal layer for treatment prediction network
        if self.training:
            lamb = (2 / (1 + math.exp(-10 * (self.current_epoch + 1)))) - 1
            lamb = torch.tensor(lamb, requires_grad=False)
        else:
            lamb = torch.tensor(0, requires_grad=False)
        a_hat = self.reversal(repr, lamb)
        # Treatment prediction network
        a_hat = fctnl.elu(self.headalayer1(a_hat))
        a_hat = torch.squeeze(torch.sigmoid(self.headalayer3(a_hat)))

        return y_hat, [h_t, c_t], a_hat

    # Model Training
    def training_step(self, train_batch, batch_idx):
        self.train()
        input_fwd = self.format_input(train_batch)
        y_data = input_fwd[1]
        a_data = input_fwd[2]
        # Forward pass
        y_hat, _, a_hat = self.forward(input_fwd)
        # Loss
        loss, y_loss, a_loss = obj_crn(a_hat, a_data, y_hat, y_data, self.alpha)

        if not self.tune_mode:
            # Logging
            self.log('train_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    # Model validation
    def validation_step(self, train_batch, batch_idx):
        self.eval()
        input_fwd = self.format_input(train_batch)
        y_data = input_fwd[1]
        a_data = input_fwd[2]
        # Forward pass
        y_hat, _, a_hat = self.forward(input_fwd)
        # Loss
        if not self.tune_mode:
            loss, y_loss, a_loss = obj_crn(a_hat, a_data, y_hat, y_data, self.alpha)
        else:
            loss = torch.mean((y_hat - y_data) ** 2)
        # Logging
        self.log('val_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

#Decoder---------------------------------------------------
# Training procedure for decoder
def train_Decoder_CRN(config, data, encoder, epochs=100, alpha=1, tune_mode=False):
    # Data
    d_train, d_val = train_test_split(data, test_size=0.2, shuffle=False)
    d_train = torch.from_numpy(d_train.astype(np.float32))
    d_val = torch.from_numpy(d_val.astype(np.float32))
    train_loader = DataLoader(dataset=d_train, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(dataset=d_val, batch_size=config["batch_size"], shuffle=False)
    # Model
    dec_crn = Decoder_crn(config=config, encoder=encoder, alpha=alpha)
    dec_crn.set_tune_mode(tune_mode)
    # Train
    Trainer1 = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False)
    Trainer1.fit(dec_crn, train_loader, val_loader)
    # Validation error after training
    val_results = Trainer1.validate(model=dec_crn, dataloaders=val_loader, verbose=False)
    val_err = val_results[0]['val_loss']
    return dec_crn, val_err


# Decoder model
class Decoder_crn(general.Causal_Decoder):
    def __init__(self, config, encoder, alpha=1):
        super().__init__(config, encoder,input_size=2)
        self.body1 = nn.Linear(config["hidden_size_lstm"], config["hidden_size_body"])
        # Outcome predictor
        self.headylayer1 = nn.Linear(config["hidden_size_body"] + 1, config["hidden_size_head"])
        self.headylayer2 = nn.Linear(config["hidden_size_head"], config["hidden_size_head"])
        self.headylayer3 = nn.Linear(config["hidden_size_head"], 1)
        self.reversal = RevGrad()
        self.headalayer1 = nn.Linear(config["hidden_size_body"], config["hidden_size_head"])
        self.headalayer2 = nn.Linear(config["hidden_size_head"], config["hidden_size_head"])
        self.headalayer3 = nn.Linear(config["hidden_size_head"], 1)
        # Optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        # Indicator whether representation balancing is applied
        self.alpha = alpha

    def forward(self, input_fwd, tf=True):
        repr_enc = input_fwd[0]
        y_hat_enc = input_fwd[1]
        A = input_fwd[2]
        y_data = input_fwd[3]
        n = y_data.size(0)
        T = y_data.size(1)
        # Memory adapter -> hidden state initialization from encoder
        h = fctnl.elu(self.adapter1(repr_enc[0]))
        c = fctnl.elu(self.adapter2(repr_enc[1]))

        y_hat = torch.zeros(n, T - 1)
        a_hat = torch.zeros(n, T - 1)
        for t in range(T-1):
        # check teacher forcing
            if tf:
                # teacher forcing
                lstm_input = torch.concat((torch.unsqueeze(A[:, t:(t+1)], 2), torch.unsqueeze(y_data[:, t:(t+1)], 2)), 2)
            else:
                #Use previous outcome predictions as input for every time step
                if t == 0:
                    lstm_input = torch.concat(
                        (torch.unsqueeze(A[:, 0:1], 2), torch.unsqueeze(torch.unsqueeze(y_hat_enc, 1), 2)), 2)
                else:
                    lstm_input = torch.concat(
                        (torch.unsqueeze(A[:, t:(t+1)], 2), torch.unsqueeze(y_hat[:, (t-1):t], 2)), 2)
            lstm_out_t, (h, c) = self.lstm(lstm_input, (h, c))
            #Build representations
            repr_a_t = fctnl.elu(self.body1(lstm_out_t))
            repr_y_t = torch.concat((repr_a_t, torch.unsqueeze(A[:, t+1:t+2], 2)), 2)
            # Outcome prediction network
            y_hat_t = fctnl.elu(self.headylayer1(repr_y_t))
            y_hat_t = torch.squeeze(self.headylayer3(y_hat_t))
            # Gradient reversal layer for treatment prediction network
            if self.training:
                lamb = (2 / (1 + math.exp(-10 * (self.current_epoch + 1)))) - 1
                lamb = torch.tensor(lamb, requires_grad=False)
            else:
                lamb = torch.tensor(0, requires_grad=False)
            #Treatment prediction head
            a_hat_t = self.reversal(repr_a_t, lamb)
            a_hat_t = fctnl.elu(self.headalayer1(a_hat_t))
            a_hat_t = torch.squeeze(torch.sigmoid(self.headalayer3(a_hat_t)))

            #Save
            y_hat[:, t] = y_hat_t
            a_hat[:, t] = a_hat_t
        return y_hat, a_hat

    # Model Training
    def training_step(self, train_batch, batch_idx):
        input_fwd = self.format_input(train_batch)
        y_data = input_fwd[2]
        a_data = input_fwd[3]
        # Forward pass
        y_hat, a_hat = self.forward(input_fwd)
        # Loss
        loss, y_loss, a_loss = obj_crn(a_hat, a_data[:, 1:], y_hat, y_data[:, 1:], self.alpha)
        if not self.tune_mode:
            # Logging
            self.log('train_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)

        return loss

    # Model validation
    def validation_step(self, train_batch, batch_idx):
        input_fwd = self.format_input(train_batch)
        y_data = input_fwd[2]
        a_data = input_fwd[3]
        # Forward pass
        y_hat, a_hat = self.forward(input_fwd)
        # Loss
        if not self.tune_mode:
            loss, loss_y, loss_a = obj_crn(a_hat, a_data[:, 1:], y_hat, y_data[:, 1:], self.alpha)
        else:
            loss = torch.mean((y_hat - y_data[:, 1:]) ** 2)
        # Logging
        self.log('val_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

def train_enc_dec_joint(config_enc, config_dec, data, epochs=100, alpha=1):
    enc_crn, _ = train_Encoder_CRN(config_enc, data, epochs=epochs, alpha=alpha)
    dec_crn, _ = train_Decoder_CRN(config_dec, data, enc_crn, epochs=epochs, alpha=alpha)
    return enc_crn, dec_crn