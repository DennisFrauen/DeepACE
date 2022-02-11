import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from abc import ABC, abstractmethod
from torch.nn.utils.rnn import PackedSequence
from typing import *
import models.helpers as helpers

class VariationalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.0):
        super().__init__()

        self.lstm_layers = [nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)]
        if num_layers > 1:
            self.lstm_layers += [nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
                                 for _ in range(num_layers - 1)]
        self.lstm_layers = nn.ModuleList(self.lstm_layers)

        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def forward(self, x, init_states=None):
        for lstm_cell in self.lstm_layers:

            # Customised LSTM-cell for variational LSTM dropout (Tensorflow-like implementation)
            if init_states is None:  # Encoder - init states are zeros
                hx = torch.zeros((x.shape[0], self.hidden_size)).type_as(x)
                cx = torch.zeros((x.shape[0], self.hidden_size)).type_as(x)
            else:  # Decoder init states are br of encoder
                hx, cx = init_states[0][0,:,:], init_states[1][0,:,:]

            # Variational dropout - sampled once per batch
            out_dropout = torch.bernoulli(hx.data.new(hx.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
            h_dropout = torch.bernoulli(hx.data.new(hx.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
            c_dropout = torch.bernoulli(cx.data.new(cx.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)

            output = []
            for t in range(x.shape[1]):
                hx, cx = lstm_cell(x[:, t, :], (hx, cx))
                if lstm_cell.training:
                    out = hx * out_dropout
                    hx, cx = hx * h_dropout, cx * c_dropout
                else:
                    out = hx
                output.append(out)

            x = torch.stack(output, dim=1)

        return x, (torch.unsqueeze(hx, 0), torch.unsqueeze(cx, 0))

#Variational dropout LSTM
class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x


class LSTM_VarDrop(nn.LSTM):
    def __init__(self, *args, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state



#General encoder model
#Neet to implement: forward(fwd_input) --> fwd output =[y_hat, repr = [h,c], misc]
#                   format_input(data_torch) --> fwd_input
#                   training_step, validation_step
class Causal_Encoder(pl.LightningModule, ABC):
    def __init__(self, config, input_size):
        super().__init__()
        #LSTM with variational dropout
        self.lstm = VariationalLSTM(input_size=input_size, hidden_size=config["hidden_size_lstm"], num_layers=1,
                                    dropout_rate=config["dropout"])
        self.tune_mode = False

    def configure_optimizers(self):
        return self.optimizer

    def set_tune_mode(self, tune_mode):
        self.tune_mode = tune_mode

    @abstractmethod
    def format_input(self, data_torch):
        pass

    @abstractmethod
    def forward(self, input_fwd):
        pass

    @abstractmethod
    def training_step(self, train_batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, train_batch, batch_idx):
        pass

    # Prediction (using numpy/torch input, numpy output)
    def predict_1step(self, data, a_int = None, y_scaler=None, output_type="np"):
        self.eval()
        #Convert to torch tensor
        if not torch.is_tensor(data):
            data = torch.from_numpy(data.astype(np.float32))
            if a_int is not None:
                a_int = torch.from_numpy(a_int(np.float32))
        n = data.size[0]
        T = data.size[1]
        if a_int is not None:
            data[:, -1, 1] = torch.squeeze(torch.full((n,1), a_int.item()))
        input_fwd = self.format_input(data)
        # Compute history representation and one-step ahead prediction with encoder
        output_fwd = self.forward(input_fwd)

        #outcome
        y_hat = output_fwd[0]
        #Representation
        [h_t, c_t] = output_fwd[1]
        if output_type == "np":
            y_hat = y_hat.detach().numpy()
            h_t = h_t.detach().numpy()
            c_t = c_t.detach().numpy()
        # Rescale y
        if y_scaler is not None:
            y_vec_scaled = np.reshape(y_hat.copy(), (n * T, 1))
            y_vec_unscaled = y_scaler.inverse_transform(y_vec_scaled)
            y_hat = np.reshape(y_vec_unscaled, (n, T))
        #Return
        if len(output_fwd) > 2:
            if output_type == "np":
                return y_hat, [h_t, c_t], output_fwd[2].detach().numpy()
            else:
                return y_hat, [h_t, c_t], output_fwd[2]
        else:
            return y_hat, [h_t, c_t]



class Causal_Decoder(pl.LightningModule, ABC):
    def __init__(self, config, encoder, input_size):
        super().__init__()
        self.encoder = encoder
        #Memory adapter
        self.adapter1 = nn.Linear(encoder.lstm.hidden_size, config["hidden_size_lstm"])
        self.adapter2 = nn.Linear(encoder.lstm.hidden_size, config["hidden_size_lstm"])
        #LSTM with variational dropout
        self.lstm = VariationalLSTM(input_size=input_size, hidden_size=config["hidden_size_lstm"], num_layers=1,
                                    dropout_rate=config["dropout"])
        self.tune_mode = False
    def configure_optimizers(self):
        return self.optimizer

    #Returns [repr_enc, y_hat_enc, A_data, y_data]
    def format_input(self, data_torch):
        self.encoder.eval()
        input_enc = self.encoder.format_input(data_torch[:, 0:1, :])
        output_enc = self.encoder.forward(input_enc)
        repr_enc = output_enc[1]
        y_hat_enc = output_enc[0]
        A_data = data_torch[:, :, 1]
        y_data = data_torch[:, :, 0]
        return [repr_enc, y_hat_enc, A_data, y_data]

    def set_tune_mode(self, tune_mode):
        self.tune_mode = tune_mode

    @abstractmethod
    def forward(self, input_fwd):
        pass

    @abstractmethod
    def training_step(self, train_batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, train_batch, batch_idx):
        pass

    # Prediction (using numpy input, numpy output)
    def estimate_ace(self, data, a_int1, a_int2, y_scaler=None):
        self.eval()
        n = data.shape[0]

        data_1 = data.copy()
        data_2 = data.copy()
        data_1[:, :, 1] = np.tile(a_int1, (n, 1))
        data_2[:, :, 1] = np.tile(a_int2, (n, 1))

        #Format input
        data1_torch = torch.from_numpy(data_1.astype(np.float32))
        data2_torch = torch.from_numpy(data_2.astype(np.float32))
        input_fwd1 = self.format_input(data1_torch)
        input_fwd2 = self.format_input(data2_torch)
        #Forward pass
        output1 = self.forward(input_fwd1)
        output2 = self.forward(input_fwd2)

        #Final predicted outcomes
        y_T_1 = output1[0][:, -1].detach().numpy()
        y_T_2 = output2[0][:, -1].detach().numpy()

        if y_scaler is not None:
            y_T_1 = helpers.rescale_y(y_T_1, y_scaler)
            y_T_2 = helpers.rescale_y(y_T_2, y_scaler)

        return np.mean(y_T_1 - y_T_2)