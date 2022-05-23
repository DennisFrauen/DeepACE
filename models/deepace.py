import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as fctnl
from sklearn.model_selection import train_test_split
import models.general_models as general
import pytorch_lightning as pl
import models.helpers as helpers


#Objective
def obj_deepace(a_hat, a_data, q_hat_f, q_hat_eps, q_hat_cf, y_data, alpha, beta):
    T = a_data.size(1)
    batch_size = a_data.size(0)
    treat_loss = fctnl.binary_cross_entropy(a_hat, a_data, reduction='mean')
    outcome_loss_M = torch.zeros((batch_size, T))
    outcome_loss_M[:, 0] = (q_hat_f[:, -1] - y_data) ** 2
    target_loss_M = torch.zeros((batch_size, T))
    target_loss_M[:, 0] = (q_hat_eps[:, -1] - y_data) ** 2
    #ytloss = torch.mean((q_hat_f[:, -1] - y_data) ** 2)
    for t in range(T - 1):
        outcome_loss_M[:, t+1] = (q_hat_f[:, t] - q_hat_cf[:, t+1]) ** 2
        target_loss_M[:, t+1] = (q_hat_eps[:, t] - q_hat_eps[:, t + 1]) ** 2
    outcome_loss = torch.mean(outcome_loss_M)
    target_loss = torch.mean(target_loss_M)
    return outcome_loss + alpha * treat_loss + beta * target_loss, outcome_loss, treat_loss, target_loss


# Encoder------------------------------------------------------------
# Training procedure for encoder
def train_deepace(config, data, a_int, epochs=100, alpha=1, beta=1, tune_mode=False):
    # Data
    d_train, d_val = train_test_split(data, test_size=0.2, shuffle=False)
    d_train = torch.from_numpy(d_train.astype(np.float32))
    d_val = torch.from_numpy(d_val.astype(np.float32))
    train_loader = DataLoader(dataset=d_train, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(dataset=d_val, batch_size=config["batch_size"], shuffle=False)
    # Model
    deepace = DeepACE(config=config, input_size=d_train.size(2) - 1, a_int=a_int, alpha=alpha, beta=beta)
    deepace.set_tune_mode(tune_mode)
    # Train
    if not tune_mode:
        #neptune_logger = NeptuneLogger(project='dennisfrauen/SeqDragonNet')
        Trainer1 = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False)# logger=neptune_logger)
    else:
        Trainer1 = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False)
    Trainer1.fit(deepace, train_loader, val_loader)
    # Validation error after training
    val_results = Trainer1.validate(model=deepace, dataloaders=val_loader, verbose=False)
    val_err = val_results[0]['val_loss']
    return deepace, val_err


# Model
class DeepACE(general.Causal_Encoder):
    def __init__(self, config, input_size, a_int, alpha=1, beta=1):
        super().__init__(config, input_size)
        self.body1 = nn.Linear(config["hidden_size_lstm"], config["hidden_size_body"])
        # Heads
        self.propnet_1 = nn.Linear(config["hidden_size_body"], config["hidden_size_head"])
        self.propnet_2 = nn.Linear(config["hidden_size_head"], 1)
        self.Qhat_1 = nn.ModuleList()
        self.Qhat_2 = nn.ModuleList()
        self.propnets_1 = nn.ModuleList()
        self.propnets_2 = nn.ModuleList()
        self.target = TargetingLayer()
        for t in range(a_int.shape[0]):
            self.Qhat_1.append(nn.Linear(config["hidden_size_lstm"] + 1, config["hidden_size_head"]))
            self.Qhat_2.append(nn.Linear(config["hidden_size_head"], 1))
            self.propnets_1.append(nn.Linear(config["hidden_size_lstm"], config["hidden_size_head"]))
            self.propnets_2.append(nn.Linear(config["hidden_size_head"], 1))
        #Dropout
        self.dropout = nn.Dropout(p=0)#config["dropout"])
        # Optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        # Intervention sequence
        self.a_int = a_int
        # Regularization parameter
        self.alpha = alpha
        self.beta = beta

    def format_input(self, data_torch):
        T = data_torch.size(1)
        t = T - self.a_int.shape[0]
        # train_batch is of size (batch_size, seq_len,input_size)
        y_data = data_torch[:, -1, 0]
        a_data = data_torch[:, :, 1]
        x_data = data_torch[:, :, 2:]
        a_input = torch.empty((data_torch.size(0), T, 1))
        a_input[:, 0, 0] = torch.zeros(data_torch.size(0))
        a_input[:, 1:, 0] = a_data[:, 0:(T - 1)]
        enc_input = torch.cat((a_input, x_data), 2)
        return enc_input, y_data, a_data[:, t:]

    def forward(self, input_fwd):
        x = input_fwd[0]
        batch_size = x.size(0)
        a_data = input_fwd[2]
        a_int_torch = torch.from_numpy(self.a_int.astype(np.float32))
        T = a_int_torch.size(0)
        # x is of size (batch_size, seq_len,input_size), treat_ind is of size (batch_size, seq_len)
        h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size)
        #factual lstm output
        lstm_out, _ = self.lstm(x, (h0, c0))
        repr = lstm_out
        #Counterfactual lstm output
        x_cf = torch.clone(x).detach()
        x_cf[:, 1:, 0] = torch.tile(torch.unsqueeze(a_int_torch[0:T-1], 0), (batch_size, 1))
        x_cf[:, 0, 0] = torch.zeros(batch_size)
        lstm_out_cf, _ = self.lstm(x_cf, (h0, c0))
        repr_cf = lstm_out_cf

        g_hat = torch.empty((batch_size, T))
        #g_hat = fctnl.elu(self.dropout(self.propnet_1(repr)))
        #g_hat = torch.squeeze(torch.sigmoid(self.propnet_2(g_hat)))
        q_hat_f = torch.empty((batch_size, T))
        q_hat_cf = torch.empty((batch_size, T))

        #Heads
        for t in range(T):
            # Propensity head
            g_hat_t = fctnl.elu(self.dropout(self.propnets_1[t](repr[:, t, :])))
            g_hat[:, t] = torch.squeeze(torch.sigmoid(self.propnets_2[t](g_hat_t)))

            # Factual Q_hat
            repr_f_t = torch.concat((repr[:, t, :], torch.unsqueeze(a_data[:, t], 1)), 1)
            q_hat_f_t = fctnl.elu(self.dropout(self.Qhat_1[t](repr_f_t)))
            q_hat_f[:, t] = torch.squeeze(self.Qhat_2[t](q_hat_f_t))

            # Counterfactual Q_hat
            a_int_t = torch.full((batch_size,1), self.a_int[t].item())
            repr_cf_t = torch.concat((repr_cf[:, t, :], a_int_t), 1)
            q_hat_cf_t = fctnl.elu(self.dropout(self.Qhat_1[t](repr_cf_t)))
            q_hat_cf[:, t] = torch.squeeze(self.Qhat_2[t](q_hat_cf_t))

        #Targeting layer
        q_hat_eps = self.target(q_hat_cf, g_hat, a_data, a_int_torch)

        # Block gradient backpropagation
        q_hat_cf_nograd = q_hat_cf.detach()
        return q_hat_f, q_hat_eps, q_hat_cf_nograd, g_hat


    # Model Training
    def training_step(self, train_batch, batch_idx):
        self.train()
        input_fwd = self.format_input(train_batch)
        y_data = input_fwd[1]
        a_data = input_fwd[2]
        # Forward pass
        q_hat_f, q_hat_eps, q_hat_cf, a_hat = self.forward(input_fwd)
        # Loss
        lamb = 0
        if self.beta > 0:
            lamb = (2/(1+math.exp(-10*(self.current_epoch+1))))-1
            lamb = torch.tensor(lamb, requires_grad=False)
        loss, y_loss, a_loss, loss_target = obj_deepace(a_hat, a_data, q_hat_f, q_hat_eps, q_hat_cf, y_data,
                                                        self.alpha, self.beta * lamb)
        if not self.tune_mode:
            # Logging
            self.log('train_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
            self.log('train_loss_a', a_loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
            self.log('train_loss_y', y_loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
            self.log('train_loss_target', loss_target.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    # Model validation
    def validation_step(self, train_batch, batch_idx):
        self.eval()
        input_fwd = self.format_input(train_batch)
        y_data = input_fwd[1]
        a_data = input_fwd[2]
        # Forward pass
        q_hat_f, q_hat_eps, q_hat_cf, a_hat = self.forward(input_fwd)
        # Loss
        lamb = 0
        if self.beta > 0:
            lamb = (2/(1+math.exp(-10*(self.current_epoch+1))))-1
            lamb = torch.tensor(lamb, requires_grad=False)
        if not self.tune_mode:
            loss, loss_y, loss_a, loss_target = obj_deepace(a_hat, a_data, q_hat_f, q_hat_eps, q_hat_cf, y_data,
                                                            self.alpha, self.beta * lamb)
            # Logging
            self.log('val_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
            self.log('val_loss_a', loss_a.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
            self.log('val_loss_y', loss_y.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
            self.log('val_loss_target', loss_target.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        else:
            loss, loss_y, loss_a, loss_target = obj_deepace(a_hat, a_data, q_hat_f, q_hat_eps, q_hat_cf, y_data, 0, 0)
            self.log('val_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)

        return loss

    def estimate_avg_outcome(self, d_train_seq, y_scaler=None, dropout=False):
        if dropout==False:
            self.eval()
        else:
            self.train()
        d_train_torch = torch.from_numpy(d_train_seq.astype(np.float32))
        # Forward pass
        input_fwd = self.format_input(d_train_torch)
        q_hat_f, q_hat_eps, q_hat_cf, a_hat = self.forward(input_fwd)
        out = q_hat_eps
        #if self.beta > 0:
        #    out = q_hat_eps
        #Rescale
        out_np = out.detach().numpy()
        out_np_unscaled = helpers.rescale_y(out_np, y_scaler)
        #Estimate
        y_avg = np.mean(out_np_unscaled[:,0])

        return y_avg


class TargetingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        #Init epsilon parameter
        self.epsilon = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.epsilon)

    def forward(self, Q_hat, g_hat, a_data, a_int):
        T = Q_hat.size(1)
        batch_size = Q_hat.size(0)
        #Calculate propensity products
        mask = a_data[:, 0] == torch.squeeze(torch.full((batch_size, 1), a_int[0]))
        indicator = torch.masked_fill(torch.zeros(batch_size), mask, 1)
        g_hat_prod = [(torch.ones(batch_size)/g_hat[:, 0])* indicator]

        for t in range(1, T):
            mask = a_data[:, t] == torch.squeeze(torch.full((batch_size, 1), a_int[t]))
            indicator = torch.masked_fill(torch.zeros(batch_size), mask, 1)
            g_hat_prod.append(g_hat_prod[t-1] * (torch.ones(batch_size)/g_hat[:, t])* indicator)

        #Calculate q's and targeted outputs
        q = [torch.zeros(batch_size)]
        Q_hat_eps = torch.empty((batch_size, T))
        #Calculate Q_epsilons
        for t in range(T-1, -1, -1):
            q.append(q[-1] - g_hat_prod[t])
            Q_hat_eps[:, t] = Q_hat[:, t] + self.epsilon * q[-1]

        return Q_hat_eps


