import numpy as np
import sklearn
from econml.orf import DMLOrthoForest, DROrthoForest
from econml.dml import CausalForestDML, LinearDML, KernelDML, NonParamDML
from econml.dr import LinearDRLearner, ForestDRLearner, DRLearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as fctnl
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import models.helpers as helpers

#Format longitudinal data (n, T, p+2) to static data History X (n, T*p), A(n, T), Y(n, 1)
def format_data(data_seq):
    n = data_seq.shape[0]
    T = data_seq.shape[1]
    p = data_seq.shape[2] - 2
    #Create static history
    hist = np.zeros((n, T*p))
    for t in range(T):
        hist[:, t*p:(t+1)*p] = data_seq[:, t, 2:]
    A = data_seq[:, :, 1]
    Y = data_seq[:, -1, 0]
    return Y, A, hist

def ace_econml(data_seq, a_int_1, a_int_2, method, y_scaler=None):
    Y_f, A_f, X_f = format_data(data_seq)
    A_cf1 = np.tile(np.expand_dims(a_int_1, 0), (data_seq.shape[0], 1))
    A_cf2 = np.tile(np.expand_dims(a_int_2, 0), (data_seq.shape[0], 1))
    if method == "dml_forest":
        est = CausalForestDML()
        est.fit(Y_f, A_f, X=X_f, W=None)
        ace = est.ate(X=X_f, T0=A_cf1, T1=A_cf2) * y_scaler.scale_
    if method == "dr_forest":
        est = ForestDRLearner()
        est.fit(Y_f, A_f[:, -1], X=X_f, W=None)
        ace = est.ate(X=X_f, T0=A_cf1[:, -1], T1=A_cf2[:, -1]) * y_scaler.scale_
    return ace


#DragonNet/ TarNet--------------------------------------------------------------------------------------------------

#Format longitudinal data (n, T, p+2) to static data History X (n, (T-1)*(p+1)+p), A(n, 1), Y(n, 1)
def format_data_drag(data_seq):
    n = data_seq.shape[0]
    T = data_seq.shape[1]
    p = data_seq.shape[2] - 2
    #Create static history
    hist = np.zeros((n, (T-1)*(p+1)+p))
    for t in range(T-1):
        hist[:, t*(p+1):(t+1)*(p+1)] = data_seq[:, t, 1:]
    hist[:, (T-1) * (p + 1):] = data_seq[:, -1, 2:]
    A = data_seq[:, -1, 1:2]
    Y = data_seq[:, -1, 0:1]

    data_drag = np.concatenate((Y, A, hist), axis=1)

    return data_drag


#Estimate ACE with dragonnet
def ace_dragonnet(config, data_seq, a_int_1, a_int_2, alpha=1, beta=1, y_scaler=None):
    data_dn = format_data_drag(data_seq)
    dn, _ = train_dragonnet(config, data_dn, alpha, beta)
    data_seq1 = data_seq.copy()
    data_seq1[:, :, 1] = np.tile(np.expand_dims(a_int_1, 0), (data_seq.shape[0], 1))
    data_seq2 = data_seq.copy()
    data_seq2[:, :, 1] = np.tile(np.expand_dims(a_int_2, 0), (data_seq.shape[0], 1))
    data_dn1 = format_data_drag(data_seq1)
    data_dn2 = format_data_drag(data_seq2)
    ace = dn.estimate_ace(data_dn1, data_dn2, y_scaler=y_scaler)
    return ace


#Loss
def loss_dragonnet(y_hat_eps, y_hat, g_hat, y_data, a_data, alpha=1, beta=1):
    outcome_loss = torch.mean((y_hat - y_data) ** 2)
    try:
        prop_loss = fctnl.binary_cross_entropy(g_hat, a_data, reduction='mean')
    except:
        prop_loss = 0
    tar_loss = torch.mean((y_hat_eps - y_data) ** 2)
    return outcome_loss + alpha * prop_loss + beta * tar_loss, outcome_loss, prop_loss, tar_loss


#Model training
# Training procedure for encoder
def train_dragonnet(config, data, alpha=1, beta=1, epochs=100):
    # Data
    d_train, d_val = train_test_split(data, test_size=0.2, shuffle=False)
    d_train = torch.from_numpy(d_train.astype(np.float32))
    d_val = torch.from_numpy(d_val.astype(np.float32))
    train_loader = DataLoader(dataset=d_train, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(dataset=d_val, batch_size=config["batch_size"], shuffle=False)
    # Model
    dn = DragonNet(config=config, input_size=d_train.size(1) - 2, alpha=alpha, beta=beta)
    # Train
    Trainer1 = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False)
    Trainer1.fit(dn, train_loader, val_loader)
    # Validation error after training
    val_results = Trainer1.validate(model=dn, dataloaders=val_loader, verbose=False)
    val_err = val_results[0]['val_loss']
    return dn, val_err


# Model
class DragonNet(pl.LightningModule):
    def __init__(self, config, input_size, alpha, beta):
        super().__init__()
        self.body1 = nn.Linear(input_size, config["hidden_size_body"])
        self.body2 = nn.Linear(config["hidden_size_body"], config["hidden_size_body"])
        self.body3 = nn.Linear(config["hidden_size_body"], config["hidden_size_body"])
        #Propnet
        self.propnet = nn.Linear(config["hidden_size_body"], 1)
        # Heads
        self.head11 = nn.Linear(config["hidden_size_body"], config["hidden_size_head"])
        self.head12 = nn.Linear(config["hidden_size_head"], config["hidden_size_head"])
        self.head13 = nn.Linear(config["hidden_size_head"], 1)
        self.head21 = nn.Linear(config["hidden_size_body"], config["hidden_size_head"])
        self.head22 = nn.Linear(config["hidden_size_head"], config["hidden_size_head"])
        self.head23 = nn.Linear(config["hidden_size_head"], 1)

        #Epsilon-layer
        self.epsilon = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.epsilon)

        #Dropout
        self.dropout = nn.Dropout(p=config["dropout"])
        # Optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])

        #Regularization
        self.alpha = alpha
        self.beta = beta

    def configure_optimizers(self):
        return self.optimizer


    def forward(self, x, A):
        batch_size = x.size(0)
        repr = fctnl.elu(self.dropout(self.body1(x)))
        repr = fctnl.elu(self.dropout(self.body2(repr)))
        repr = fctnl.elu(self.dropout(self.body3(repr)))

        g_hat = torch.squeeze(torch.sigmoid(self.propnet(repr)))

        head1 = fctnl.elu(self.dropout(self.head11(repr)))
        head1 = fctnl.elu(self.dropout(self.head12(head1)))
        head1 = torch.squeeze(fctnl.elu(self.dropout(self.head13(head1))))

        head2 = fctnl.elu(self.dropout(self.head21(repr)))
        head2 = fctnl.elu(self.dropout(self.head22(head2)))
        head2 = torch.squeeze(fctnl.elu(self.dropout(self.head23(head2))))

        y_hat = A * head2 + (torch.ones(batch_size) - A) * head1

        #Epsilon layer
        y_hat_eps = y_hat + self.epsilon * ((A/g_hat) - ((1-A)/(1-g_hat)))

        return y_hat_eps, y_hat, g_hat


    def training_step(self, train_batch, batch_idx):
        self.train()
        y_data = train_batch[:, 0]
        a_data = train_batch[:, 1]
        x_data = train_batch[:, 2:]
        # Forward pass
        y_hat_eps, y_hat, g_hat = self.forward(x_data, a_data)
        # Loss
        loss, _, _, _ = loss_dragonnet(y_hat_eps, y_hat, g_hat, y_data, a_data, alpha=self.alpha, beta=self.beta)
        return loss

    # Model validation
    def validation_step(self, train_batch, batch_idx):
        self.eval()
        y_data = train_batch[:, 0]
        a_data = train_batch[:, 1]
        x_data = train_batch[:, 2:]
        # Forward pass
        y_hat_eps, y_hat, g_hat = self.forward(x_data, a_data)
        # Loss
        _, outcome_loss, _, _ = loss_dragonnet(y_hat_eps, y_hat, g_hat, y_data, a_data, alpha=self.alpha, beta=self.beta)
        self.log('val_loss', outcome_loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return outcome_loss

    def estimate_ace(self, data_cf1, data_cf2, y_scaler=None):
        self.eval()
        data_torch1 = torch.from_numpy(data_cf1.astype(np.float32))
        data_torch2 = torch.from_numpy(data_cf2.astype(np.float32))

        a_data1 = data_torch1[:, 1]
        a_data2 = data_torch2[:, 1]
        x_data1 = data_torch1[:, 2:]
        x_data2 = data_torch2[:, 2:]
        y_hat_eps1, _, _ = self.forward(x_data1, a_data1)
        y_hat_eps2, _, _ = self.forward(x_data2, a_data2)
        #Rescale
        y_1 = y_hat_eps1.detach().numpy()
        y_2 = y_hat_eps2.detach().numpy()
        y_1_unscaled = helpers.rescale_y(y_1, y_scaler)
        y_2_unscaled = helpers.rescale_y(y_2, y_scaler)
        #Estimate
        ace = np.mean(y_1_unscaled - y_2_unscaled)

        return ace