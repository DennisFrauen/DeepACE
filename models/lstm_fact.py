import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import models.helpers as helpers
from sklearn.model_selection import train_test_split
import models.general_models as general
import pytorch_lightning as pl
import torch.nn.functional as fctnl

#Simple factual lstm--------------------------------------

#Factual msn objective
def obj_lstm(y_hat, y_data):
    outcome_loss = torch.mean((y_hat - y_data) ** 2)
    return outcome_loss

# Training procedure
def train_lstm_fact(config, data, epochs=100, tune_mode=False):
    # Data
    d_train, d_val = train_test_split(data, test_size=0.2, shuffle=False)
    d_train = torch.from_numpy(d_train.astype(np.float32))
    d_val = torch.from_numpy(d_val.astype(np.float32))
    train_loader = DataLoader(dataset=d_train, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(dataset=d_val, batch_size=config["batch_size"], shuffle=False)
    # Model
    lstm_f = LSTM_fact(config=config, input_size=2, p=(d_train.size(2)-2))
    lstm_f.set_tune_mode(tune_mode)
    # Train
    Trainer1 = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False)
    Trainer1.fit(lstm_f, train_loader, val_loader)
    # Validation error after training
    val_results = Trainer1.validate(model=lstm_f, dataloaders=val_loader, verbose=False)
    val_err = val_results[0]['val_loss']
    return lstm_f, val_err


# Model
class LSTM_fact(general.Causal_Encoder):
    def __init__(self, config, input_size, p):
        super().__init__(config, input_size)
        #Input adapter- transforms baselinecovariates to input
        self.adapter = nn.Linear(p+1, config["hidden_size_lstm"]*2)
        # Outcome predictor
        self.output = nn.Linear(config["hidden_size_lstm"], 1)
        # Optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])


    def format_input(self, data_torch):
        y_data = data_torch[:, :, 0]
        enc_input = data_torch[:, :, 1:]
        return enc_input, y_data


    def forward(self, input_fwd, tf=True):
        x = input_fwd[0]
        A = x[:, 1:, 0]
        y_data = input_fwd[1]#x[:, 1:, 0]
        n = x.size(0)
        T = x.size(1)
        input_repr = fctnl.elu(self.adapter(x[:, 0, :]))
        # x is of size (batch_size, seq_len,input_size), treat_ind is of size (batch_size, seq_len)
        h = torch.unsqueeze(input_repr[:, 0:self.lstm.hidden_size], 0)
        c = torch.unsqueeze(input_repr[:, self.lstm.hidden_size:], 0)

        y_hat = torch.zeros(n, T - 1)
        for t in range(T - 1):
            # check teacher forcing
            if tf:
                # teacher forcing
                lstm_input = torch.concat((torch.unsqueeze(A[:, t:(t + 1)], 2), torch.unsqueeze(y_data[:, t:(t + 1)], 2)), 2)
            else:
                # Use previous outcome predictions as input for every time step
                if t == 0:
                    lstm_input = torch.concat(
                        (torch.unsqueeze(A[:, 0:1], 2), torch.unsqueeze(y_hat[:, 0:1], 2)), 2)
                else:
                    lstm_input = torch.concat(
                        (torch.unsqueeze(A[:, t:(t + 1)], 2), torch.unsqueeze(y_hat[:, (t - 1):t], 2)), 2)
            lstm_out_t, (h, c) = self.lstm(lstm_input, (h, c))
            # Save
            y_hat[:, t] = torch.squeeze(self.output(lstm_out_t))

        return y_hat

    # Model Training
    def training_step(self, train_batch, batch_idx):
        self.train()
        input_fwd = self.format_input(train_batch)
        y_data = input_fwd[1]
        # Forward pass
        y_hat = self.forward(input_fwd)
        # Loss
        loss = obj_lstm(y_hat, y_data[:, 1:])
        return loss

    # Model validation
    def validation_step(self, train_batch, batch_idx):
        self.eval()
        input_fwd = self.format_input(train_batch)
        y_data = input_fwd[1]
        # Forward pass
        y_hat = self.forward(input_fwd)
        # Loss
        loss = obj_lstm(y_hat, y_data[:, 1:])
        # Logging
        self.log('val_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def estimate_ace(self, data, a_int1, a_int2, y_scaler=None):
        self.eval()
        n = data.shape[0]

        data_1 = data.copy()
        data_2 = data.copy()
        data_1[:, :, 1] = np.tile(a_int1, (n, 1))
        data_2[:, :, 1] = np.tile(a_int2, (n, 1))

        # Format input
        data1_torch = torch.from_numpy(data_1.astype(np.float32))
        data2_torch = torch.from_numpy(data_2.astype(np.float32))
        input_fwd1 = self.format_input(data1_torch)
        input_fwd2 = self.format_input(data2_torch)
        # Forward pass
        output1 = self.forward(input_fwd1)
        output2 = self.forward(input_fwd2)

        # Final predicted outcomes
        y_T_1 = output1[:, -1].detach().numpy()
        y_T_2 = output2[:, -1].detach().numpy()

        if y_scaler is not None:
            y_T_1 = helpers.rescale_y(y_T_1, y_scaler)
            y_T_2 = helpers.rescale_y(y_T_2, y_scaler)

        return np.mean(y_T_1 - y_T_2)