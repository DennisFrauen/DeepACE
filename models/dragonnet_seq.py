# Code for sequential dragonnet model
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as fctnl
from pytorch_lightning.loggers.neptune import NeptuneLogger
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import models.helpers as helpers
import models.general_models as general

# Objective function for encoder/ decoder of sequential dragonnet
# Mean squared loss (Outcomes) + alpha * crossentropy loss (propensity score)
def obj_dragon(g_hat, a_train, y_hat, y_train, alpha):
    outcome_loss = torch.mean((y_hat - y_train) ** 2)
    prop_loss = fctnl.binary_cross_entropy(g_hat, a_train, reduction='mean')
    return outcome_loss + alpha * prop_loss, outcome_loss, prop_loss


# Encoder training---------------------------------------------------------------------

#Training procedure for encoder
def train_Encoder_DragonNet(config, data, alpha, epochs=100, callbacks = []):
    # Neptune Logger
    # neptune.init(project_qualified_name='dennisfrauen/SeqDragonNet')
    neptune_logger = NeptuneLogger(project='dennisfrauen/SeqDragonNet')
    # Data
    d_train, d_val = train_test_split(data, test_size=0.2, shuffle=False)
    d_train = torch.from_numpy(d_train.astype(np.float32))
    d_val = torch.from_numpy(d_val.astype(np.float32))
    train_loader = DataLoader(dataset=d_train, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(dataset=d_val, batch_size=config["batch_size"], shuffle=False)
    # Model
    enc_dragonnet = Encoder_DragonNet(config=config, input_size=d_train.size(2) - 1, alpha=alpha)

    # Train
    Trainer1 = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, logger=neptune_logger,
                          enable_model_summary=False, callbacks= callbacks)
    Trainer1.fit(enc_dragonnet, train_loader, val_loader)
    #Validation error after training
    val_results = Trainer1.validate(model=enc_dragonnet, dataloaders=val_loader, verbose=False)
    val_err = val_results[0]['val_loss']
    return enc_dragonnet, val_err


# Encoder model
class Encoder_DragonNet(general.Causal_Encoder):
    def __init__(self, config, input_size, alpha):
        super().__init__(config, input_size)
        self.body1 = nn.Linear(config["hidden_size_lstm"], config["hidden_size_body"])
        self.prop = nn.Linear(config["hidden_size_body"], 1)
        # Dragon Heads
        self.head1layer1 = nn.Linear(config["hidden_size_body"], config["hidden_size_head"])
        self.head1layer2 = nn.Linear(config["hidden_size_head"], config["hidden_size_head"])
        self.head1layer3 = nn.Linear(config["hidden_size_head"], 1)
        self.head0layer1 = nn.Linear(config["hidden_size_body"], config["hidden_size_head"])
        self.head0layer2 = nn.Linear(config["hidden_size_head"], config["hidden_size_head"])
        self.head0layer3 = nn.Linear(config["hidden_size_head"], 1)
        # Optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
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
        treat_ind = input_fwd[2]
        # x is of size (batch_size, seq_len,input_size), treat_ind is of size (batch_size, seq_len)
        h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size)
        # lstm_out: tensor of shape (batch_size, seq_length, hidden_size_lstm) - Outputs are hidden states
        lstm_out, (h_t, c_t) = self.lstm(x, (h0, c0))
        # Dragon body
        repr = fctnl.elu(self.body1(lstm_out))
        # Middle head
        g_hat = torch.squeeze(torch.sigmoid(self.prop(repr)))
        # g_hat is of size (batch_size, seq_size)
        # outer heads
        # Head 1
        y_1_hat = fctnl.elu(self.head1layer1(repr))
        #y_1_hat = fctnl.elu(self.head1layer2(y_1_hat))
        y_1_hat = torch.squeeze(self.head1layer3(y_1_hat))
        # head 2
        y_0_hat = fctnl.elu(self.head0layer1(repr))
        #y_0_hat = fctnl.elu(self.head0layer2(y_0_hat))
        y_0_hat = torch.squeeze(self.head0layer3(y_0_hat))
        # y_1_hat and y_0_hat are of size (batch_size, seq_size)
        # Choose head for output depending on current treatment
        y_hat = treat_ind * y_1_hat + (1 - treat_ind) * y_0_hat
        return y_hat, [h_t, c_t], g_hat

    # Model Training
    def training_step(self, train_batch, batch_idx):
        input_fwd = self.format_input(train_batch)
        y_data = input_fwd[1]
        a_data = input_fwd[2]
        # Forward pass
        y_hat, _, g_hat = self.forward(input_fwd)
        # Loss
        loss, y_loss, prop_loss = obj_dragon(g_hat, a_data, y_hat, y_data, self.alpha)
        if self.alpha > 0:
            prop_loss = prop_loss * self.alpha
        # Logging
        self.log('loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('loss_prop', prop_loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('loss_y', y_loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    # Model validation
    def validation_step(self, train_batch, batch_idx):
        input_fwd = self.format_input(train_batch)
        y_data = input_fwd[1]
        a_data = input_fwd[2]
        # Forward pass
        y_hat, _, g_hat = self.forward(input_fwd)
        # Loss
        loss, y_loss, prop_loss = obj_dragon(g_hat, a_data, y_hat, y_data, self.alpha)
        # Logging
        self.log('val_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_loss_prop', prop_loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_loss_y', y_loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss


# Decoder training--------------------------------------------------------------

def train_Decoder_DragonNet(data, encoder, alpha, tau_max, p_static, hidden_size_lstm, hidden_size_body, hidden_size_head,
                            lr=5e-4):
    # Neptune Logger
    # neptune.init(project_qualified_name='dennisfrauen/SeqDragonNet')
    neptune_logger = NeptuneLogger(project='dennisfrauen/SeqDragonNet')
    n = data.shape[0]
    batch_size = int(n / 10)
    # Explode dataset
    # data expl is of shape (n*(T-tau),tau,p+2+ dim(h_t) + dim(c_t)
    # representations and encoder one-step ahead prediction for each batch/ timestep are saved in data_exp[:,0,(p+2):]
    d_train, d_val = helpers.explode_dataset(data, encoder, tau_max, batch_size)
    #Data Loaders
    train_loader = DataLoader(dataset=d_train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=d_val, batch_size=batch_size, shuffle=False)

    # Decoder model, input is treatment and previous predicted outcome -> input_size = 2
    dec_dragonnet = Decoder_DragonNet(encoder, 2+p_static, hidden_size_lstm, hidden_size_body, hidden_size_head, 0, alpha, p_static)

    # Training using ADAM
    dec_dragonnet.optimizer = torch.optim.Adam(dec_dragonnet.parameters(), lr=lr, weight_decay=0)
    # Train
    Trainer1 = pl.Trainer(max_epochs=50, logger=neptune_logger, enable_progress_bar=False, enable_model_summary=False)
    Trainer1.fit(dec_dragonnet, train_loader, val_loader)
    return dec_dragonnet


class Decoder_DragonNet(pl.LightningModule):
    def __init__(self, encoder, input_size, hidden_size_lstm, hidden_size_body, hidden_size_head, dropout, alpha, p_static):
        super().__init__()
        # Memory adapters
        self.adapter1 = nn.Linear(encoder.lstm.hidden_size, hidden_size_lstm)
        self.adapter2 = nn.Linear(encoder.lstm.hidden_size, hidden_size_lstm)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size_lstm, num_layers=1, batch_first=True)
        self.body1 = nn.Linear(hidden_size_lstm, hidden_size_body)
        self.prop = nn.Linear(hidden_size_body, 1)
        # Dragon Heads
        self.head1layer1 = nn.Linear(hidden_size_body, hidden_size_head)
        self.head1layer2 = nn.Linear(hidden_size_head, hidden_size_head)
        self.head1layer3 = nn.Linear(hidden_size_head, 1)
        self.head0layer1 = nn.Linear(hidden_size_body, hidden_size_head)
        self.head0layer2 = nn.Linear(hidden_size_head, hidden_size_head)
        self.head0layer3 = nn.Linear(hidden_size_head, 1)
        # Optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.alpha = alpha
        self.encoder = encoder
        self.p_static = p_static

    def forward(self, x, treat_ind, repr_t, y_hat_enc, tf_ratio=0.5):
        # x is of size (batch_size, seq_len,1 or 2) lagged treatments and (during training) lagged previous outcomes
        # treat_ind is of size (batch_size, seq_len)
        # y_hat_enc is of size (batch_size), is one-step ahead prediction by encoder
        n = x.size(0)
        T = x.size(1)
        input_dim = x.size(2)
        # Memory adapter -> hidden state initialization from encoder
        h = fctnl.elu(self.adapter1(repr_t[0]))
        c = fctnl.elu(self.adapter2(repr_t[1]))
        # Input for lstm of size (batch_size,T=1, 2(lagged treat + outcomes))
        lstm_in = torch.empty((n, 1, 2))
        # Tensors to save outputs
        g_hat = torch.empty((n, T))
        y_hat = torch.empty((n, T))
        # Forward pass through lstm, using teacher forcing
        for t in range(T):
            # Teacher forcing
            if np.random.random() < tf_ratio:
                lstm_in = x[:, t:(t + 1), :]
            else:
                if t == 0:
                    y_hat_enc = torch.unsqueeze(torch.unsqueeze(y_hat_enc, 1), 2)
                    lstm_in = torch.cat((x[:, t:(t + 1), 0:(self.p_static + 1)], y_hat_enc), 2)
                else:
                    # Use previous predicted outcome
                    y_hat_prev = torch.unsqueeze(y_hat[:, (t - 1):t], 2)
                    lstm_in = torch.cat((x[:, t:(t + 1), 0:(self.p_static + 1)], y_hat_prev), 2)
            lstm_out, (h, c) = self.lstm(lstm_in, (h, c))
            repr = fctnl.elu(self.body1(lstm_out))
            # lstm_out: tensor of shape (batch_size, 1, hidden_size_lstm) - Outputs are hidden states
            # Middle head
            g_hat[:, t] = torch.squeeze(torch.sigmoid(self.prop(repr)))
            # outer heads
            # Head 1
            y_1_hat = fctnl.elu(self.head1layer1(repr))
            # y_1_hat = fctnl.elu(self.head1layer2(y_1_hat))
            y_1_hat = torch.squeeze(self.head1layer3(y_1_hat))
            # head 2
            y_0_hat = fctnl.elu(self.head0layer1(repr))
            # y_0_hat = fctnl.elu(self.head0layer2(y_0_hat))
            y_0_hat = torch.squeeze(self.head0layer3(y_0_hat))
            # y_1_hat and y_0_hat are of size (batch_size, seq_size)
            # Choose head for output depending on current treatment
            y_hat[:, t] = treat_ind[:, t] * y_1_hat + (1 - treat_ind[:, t]) * y_0_hat

        return g_hat, y_hat

    def configure_optimizers(self):
        return self.optimizer

    # Model Training
    def training_step(self, train_batch, batch_idx):
        dec_input, y_data, a_data, repr_enc, y_hat_enc = helpers.format_exploded_batch(train_batch, self.encoder.lstm.hidden_size, self.p_static)
        # Forward pass
        g_hat, y_hat = self.forward(dec_input, a_data, repr_enc, y_hat_enc, tf_ratio=0.5)
        # Loss
        loss, y_loss, prop_loss = obj_dragon(g_hat, a_data, y_hat, y_data, self.alpha)
        if self.alpha > 0:
            prop_loss = prop_loss * self.alpha
        # Logging
        self.log('loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('loss_prop', prop_loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('loss_y', y_loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    # Validation
    def validation_step(self, train_batch, batch_idx):
        dec_input, y_data, a_data, repr_enc, y_hat_enc = helpers.format_exploded_batch(train_batch, self.encoder.lstm.hidden_size, self.p_static)
        # Forward pass
        g_hat, y_hat = self.forward(dec_input, a_data, repr_enc, y_hat_enc)
        # Loss
        loss, y_loss, prop_loss = obj_dragon(g_hat, a_data, y_hat, y_data, self.alpha)
        if self.alpha > 0:
            prop_loss = prop_loss * self.alpha
        # Logging
        self.log('val_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_loss_prop', prop_loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_loss_y', y_loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    # Prediction (using numpy/torch input, numpy output)
    #X_static is of shape (n,p_static)
    #A_int is of shape(n,tau_dec)
    def predict(self, A_int, repr_enc, y_hat_enc, X_static, y_scaler=None):
        T = A_int.shape[1]
        n = y_hat_enc.shape[0]
        #transform static covariates to decoder input
        X_static_input = np.empty((n, T-1, self.p_static))
        for t in range(T-1):
            X_static_input[:,t,:] = X_static
        X_static_input = torch.from_numpy(X_static_input.astype(np.float32))
        # Interventions as input for decoder
        a_input = A_int[:,:T - 1]
        a_input = torch.from_numpy(a_input.astype(np.float32))
        a_input = torch.unsqueeze(a_input, 2)
        #Final decoder input
        dec_input = torch.concat((a_input,X_static_input),dim=2)
        # Treat indices for head choice
        a_data = A_int[:,1:T]
        a_data = torch.from_numpy(a_data.astype(np.float32))
        h_t = torch.from_numpy(repr_enc[0].astype(np.float32))
        c_t = torch.from_numpy(repr_enc[1].astype(np.float32))
        # Compute history representation and one-step ahead prediction with encoder
        g_hat, y_hat = self.forward(dec_input, a_data, [h_t, c_t],
                                    torch.from_numpy(y_hat_enc.astype(np.float32)), tf_ratio=0)
        g_hat = g_hat.detach().numpy()
        y_hat = y_hat.detach().numpy()
        # Rescale y
        if y_scaler is not None:
            y_vec_scaled = np.reshape(y_hat.copy(), (n * (T - 1), 1))
            y_vec_unscaled = y_scaler.inverse_transform(y_vec_scaled)
            y_hat = np.reshape(y_vec_unscaled, (n, (T - 1)))
        return g_hat, y_hat

    # Joint prediction with encoder and decoder
    def joint_prediction(self, d_train_seq, a_int, tau_enc=1, y_scaler=None):
        #Scaled predictions of encoder
        g_hat_enc, y_hat_enc, _ = self.encoder.predict_1step(d_train_seq,a_int[:tau_enc], y_scaler)
        #Unscaled predictions of encoder
        _, y_hat_enc_us, repr_enc = self.encoder.predict_1step(d_train_seq, a_int[:tau_enc])
        # One-step-ahead prediction encoder
        y_hat_t = y_hat_enc_us[:, -1]
        #Decoder prediction
        g_hat_dec, y_hat_dec = self.predict(a_int[tau_enc - 1:], repr_enc, y_hat_t, y_scaler)
        #Joint prediction
        g_hat = np.concatenate((g_hat_enc, g_hat_dec), 1)
        y_hat = np.concatenate((y_hat_enc, y_hat_dec), 1)
        return g_hat, y_hat

