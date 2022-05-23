
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as fctnl
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
import plotting.plots
from sklearn.model_selection import train_test_split



def obj_dragon(prop_est, prop_true, y_est, y_true, reg):
    outcome_loss = torch.mean((y_est - y_true) ** 2)

    prop_loss = fctnl.binary_cross_entropy(prop_est, prop_true, reduction='mean')
    return outcome_loss + reg * prop_loss, outcome_loss, prop_loss
    #return prop_loss, outcome_loss, prop_loss

def train_DragonNet_PL(data,alpha,type='tarnet',body_size=100,head_size=50):
    #Data
    d_train, d_val = train_test_split(data,test_size=0.2,shuffle=True)
    d_train = torch.from_numpy(d_train.astype(np.float32))
    d_val = torch.from_numpy(d_val.astype(np.float32))
    train_loader = DataLoader(dataset=d_train, batch_size=100, shuffle=False)
    val_loader = DataLoader(dataset=d_val, batch_size=100, shuffle=False)
    p = d_train.shape[1] - 2
    #Model
    if type=='tarnet':
        dragonnet = TarNet_PL(p, body_size, head_size, 0, alpha)
    else:
        dragonnet = DragonNet_PL(p,body_size,head_size,0,alpha)

    #First step using ADAM
    #early_stop_callback_ADAM = EarlyStopping(monitor="epoch_loss_val", min_delta=0.00, patience=10, verbose=False, mode="min")
    dragonnet.optimizer = torch.optim.Adam(dragonnet.parameters(),lr=1e-3,weight_decay=0)
    #Train
    Trainer1 = pl.Trainer(max_epochs=100,progress_bar_refresh_rate=0,weights_summary=None)#,callbacks=[early_stop_callback_ADAM])
    Trainer1.fit(dragonnet,train_loader,val_loader)

    """
    #Second step using SGD
    early_stop_callback_SGD = EarlyStopping(monitor="epoch_loss_val", min_delta=0.00, patience=20, verbose=False, mode="min")
    dragonnet.optimizer = torch.optim.SGD(dragonnet.parameters(), lr=1e-4,momentum=0.9,nesterov=True,weight_decay=0)
    #Train
    Trainer2 = pl.Trainer(max_epochs=150,logger=neptune_logger,callbacks=[early_stop_callback_SGD])
    Trainer2.fit(dragonnet,train_loader,val_loader)
    """
    return dragonnet

class Net_PL(pl.LightningModule):
    def __init__(self,input_size, hidden_size, hidden_size_dragon, dropout,alpha):
        super().__init__()
        self.alpha = alpha
        #Dragon Body
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.hidden_size = hidden_size
        self.body1 = nn.Linear(input_size,hidden_size)
        self.drop_body1 = nn.Dropout(dropout)
        self.body2 = nn.Linear(hidden_size, hidden_size)
        self.drop_body2 = nn.Dropout(dropout)
        self.body3 = nn.Linear(hidden_size, hidden_size)
        #Heads
        self.layer11 = nn.Linear(hidden_size, hidden_size_dragon)
        self.layer12 = nn.Linear(hidden_size_dragon, hidden_size_dragon)
        self.layer13 = nn.Linear(hidden_size_dragon, 1)
        self.layer21 = nn.Linear(hidden_size, hidden_size_dragon)
        self.layer22 = nn.Linear(hidden_size_dragon, hidden_size_dragon)
        self.layer23 = nn.Linear(hidden_size_dragon, 1)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,mode='min',
        #                                                                factor=0.5,patience=5)
        #Parameter initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data,mean=0,std=0.05)
                torch.nn.init.normal_(m.bias.data,mean=0,std=0.05)
        self.apply(init_weights)


    def configure_optimizers(self):
        return self.optimizer
        """
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "epoch_loss",
                "frequency": 1
            },
        }
        """
    def training_step(self, train_batch, batch_idx):
        y_data = train_batch[:, 0]
        a_data = train_batch[:, 1]
        x_data = train_batch[:, 2:]
        g_hat, y_hat, _ = self.forward(x_data, a_data)
        # Loss
        loss, loss_y, loss_prop = obj_dragon(g_hat, a_data, y_hat, y_data, self.alpha)
        #Weight decay
        """
        l2_reg = torch.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        loss += 0.01 * l2_reg
        """
        if self.alpha > 0:
            loss_prop = loss_prop*self.alpha
        #Logging
        self.log('loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('loss_prop', loss_prop.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('loss_y', loss_y.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self,train_batch,batchidx):
        y_data = train_batch[:, 0]
        a_data = train_batch[:, 1]
        x_data = train_batch[:, 2:]
        g_hat, y_hat, _ = self.forward(x_data, a_data)
        # Loss
        loss, loss_y, loss_prop = obj_dragon(g_hat, a_data, y_hat, y_data, self.alpha)
        # Weight decay
        #l2_reg = torch.tensor(0.)
        #for param in self.parameters():
        #    l2_reg += torch.norm(param)
        #loss += 0.01 * l2_reg
        #Logging
        self.log('val_loss', loss.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_loss_prop', loss_prop.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_loss_y', loss_y.detach().numpy().item(), logger=True, on_epoch=True, on_step=False)

        return loss


    def test_eval(self, d_test,y_scaler = None):
        # Data processing for encoder input
        d_test = torch.from_numpy(d_test.astype(np.float32))
        a_val = d_test[:, 1]
        x_val = d_test[:, 2:]
        # Compute history representation and one-step ahead prediction with encoder
        g_hat, y_hat_enc, repr = self.forward(x_val, a_val)
        y_hat_enc = y_hat_enc.detach().numpy()
        #Re-Scale predicted outcomes if necessary
        if y_scaler is not None:
            y_hat_enc = y_scaler.inverse_transform(np.expand_dims(y_hat_enc,1))
            y_hat_enc = y_hat_enc[:,0]
        return g_hat.detach().numpy(),y_hat_enc, repr.detach().numpy()

    def estimate_ate(self,d_test,y_scaler=None,g_hat=None):
        n = d_test.shape[0]
        #Predict both treatment groups
        d_test_int1 = d_test.copy()
        d_test_int1[:, 1] = np.full(n,1)
        g_hat_1, y_hat_int1, _ = self.test_eval(d_test_int1,y_scaler)
        d_test_int0 = d_test.copy()
        d_test_int0[:, 1] = np.full(n,0)
        g_hat_0, y_hat_int0, _ = self.test_eval(d_test_int0,y_scaler)
        #Remove if overlapping assumption violated
        if g_hat is not None:
            keep_these = np.logical_and(g_hat >= 0.01, g_hat <= 1. - 0.01)
            y_hat_int1 = y_hat_int1[keep_these]
            y_hat_int0 = y_hat_int0[keep_these]
        # Estimated ATE
        ate_hat = np.mean(y_hat_int1 - y_hat_int0)
        #ate_hat = np.mean(y_hat_int)
        return ate_hat

class DragonNet_PL(Net_PL):
    def __init__(self, input_size, hidden_size, hidden_size_dragon, dropout, alpha):
        super().__init__(input_size, hidden_size, hidden_size_dragon, dropout, alpha)
        #Propensity estimate
        self.propnet = nn.Linear(hidden_size, 1)

    def forward(self, x,treat_ind):
        # x is of size (batch_size,input_size)
        # treat_ind is of size n
        n = x.size(0)
        #repr = self.batch_norm(x)
        repr = fctnl.elu(self.body1(x))
        repr = fctnl.elu(self.body2(repr))
        #repr = fctnl.elu(self.body3(repr))
        g_hat = torch.squeeze(torch.sigmoid(self.propnet(repr)))
        #Head 1
        y_1_hat = fctnl.elu(self.layer11(repr))
        #y_1_hat = fctnl.elu(self.layer12(y_1_hat))
        y_1_hat = torch.squeeze(self.layer13(y_1_hat))
        # head 2
        y_0_hat = fctnl.elu(self.layer21(repr))
        #y_0_hat = fctnl.elu(self.layer22(y_0_hat))
        y_0_hat = torch.squeeze(self.layer23(y_0_hat))
        #Output
        y_hat = treat_ind * y_1_hat + (torch.ones(n) - treat_ind) * y_0_hat
        return g_hat, y_hat, repr


class TarNet_PL(Net_PL):
    def __init__(self, input_size, hidden_size, hidden_size_dragon, dropout, alpha):
        super().__init__(input_size, hidden_size, hidden_size_dragon, dropout, alpha)
        # Propensity estimate
        self.propnet = nn.Linear(input_size, 1)

    def forward(self, x, treat_ind):
        # x is of size (batch_size,input_size)
        # treat_ind is of size n
        n = x.size(0)
        repr = fctnl.elu(self.body1(x))
        repr = fctnl.elu(self.body2(repr))
        repr = fctnl.elu(self.body3(repr))
        g_hat = torch.squeeze(torch.sigmoid(self.propnet(x)))
        # Head 1
        y_1_hat = fctnl.elu(self.layer11(repr))
        y_1_hat = fctnl.elu(self.layer12(y_1_hat))
        y_1_hat = torch.squeeze(self.layer13(y_1_hat))
        # head 2
        y_0_hat = fctnl.elu(self.layer21(repr))
        y_0_hat = fctnl.elu(self.layer22(y_0_hat))
        y_0_hat = torch.squeeze(self.layer23(y_0_hat))
        # Output
        y_hat = treat_ind * y_1_hat + (torch.ones(n) - treat_ind) * y_0_hat
        return g_hat, y_hat, repr

class DragonNet_Static(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(DragonNet_Static, self).__init__()
        self.hidden_size = hidden_size
        self.body1 = nn.Linear(input_size,hidden_size)
        self.drop_body1 = nn.Dropout(dropout)
        self.body2 = nn.Linear(hidden_size, hidden_size)
        self.drop_body2 = nn.Dropout(dropout)
        self.body3 = nn.Linear(hidden_size, hidden_size)
        #Propensity estimate
        self.propnet = nn.Linear(hidden_size, 1)


    def train_procedure(self,d_train,alpha, batch_size, lr,n_epochs=100,moment=0,plot_loss=False):
        n_epochs = 400
        data = torch.from_numpy(d_train.astype(np.float32))
        train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)
        optimizer_SGD = torch.optim.SGD(self.parameters(), lr=1e-5,momentum=0.9,nesterov=False,weight_decay=0.01)
        optimizer_ADAM = torch.optim.Adam(self.parameters(), lr=1e-3,weight_decay=0.01)

        # Train
        loss_over_epochs = np.zeros(n_epochs)
        loss_over_epochs_y = np.zeros(n_epochs)
        loss_over_epochs_prop = np.zeros(n_epochs)
        print('train encoder')
        for i, epoch in enumerate(range(n_epochs)):
            for batch in train_loader:
                # Process data for encoder
                y_data = batch[:, 0]
                a_data = batch[:, 1]
                x_data = batch[:,2:]
                # Forward pass
                g_hat, y_hat, _ = self.forward(x_data, a_data)
                # Loss
                loss_enc, loss_y, loss_prop = obj_dragon(g_hat, a_data, y_hat, y_data, alpha)
                # Update
                loss_enc.backward()
                if i < 100:
                    optimizer_ADAM.step()
                    optimizer_ADAM.zero_grad()
                else:
                    optimizer_SGD.step()
                    optimizer_SGD.zero_grad()
                #Safe losses for plotting
                loss_over_epochs[i] += loss_enc.item()
                loss_over_epochs_y[i] += loss_y.item()
                loss_over_epochs_prop[i] += loss_prop.item()
            #Average over batches
            loss_over_epochs[i] = loss_over_epochs[i] / len(train_loader)
            loss_over_epochs_y[i] = loss_over_epochs_y[i]/len(train_loader)
            loss_over_epochs_prop[i] = loss_over_epochs_prop[i] / len(train_loader)
            # if (epoch + 1) % n_epochs == 0:
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss_over_epochs[-1]:.4f}')
        if plot_loss == True:
            plotting.plots.plot_trajectory_general(loss_over_epochs,loss_over_epochs_y,loss_over_epochs_prop,
                                                   title='Training loss over epochs')

    def validation_step_cf(self,d_val,a_int,ate):
        ate_est = self.estimate_ate(d_val,a_int)
        err = np.absolute(ate_est - ate)
        return err

    def validation_step(self,d_val):
        _, y_hat, _ = self.test_eval(d_val)
        y_true = d_val[:, 0]
        return np.sqrt(np.sum((y_true - y_hat)**2))

    def validation_step_alpha(self,d_val,alpha):
        def cross_entropy(predictions, targets):
            N = predictions.shape[0]
            ce = -np.sum(targets * np.log(predictions)) / N
            return ce
        g_hat, y_hat, _ = self.test_eval(d_val)
        g_hat = g_hat.astype(np.float32)

        y_true = d_val[:, 0]
        a_true = d_val[:, 1].astype(np.float32)
        outcome_loss = np.sqrt(np.sum((y_true - y_hat)**2))
        prop_loss = fctnl.binary_cross_entropy(torch.from_numpy(g_hat), torch.from_numpy(a_true))
        prop_loss = prop_loss.detach().numpy()
        return outcome_loss + alpha* prop_loss

    def test_eval(self, d_test,y_scaler = None):
        # Data processing for encoder input
        d_test = torch.from_numpy(d_test.astype(np.float32))
        a_val = d_test[:, 1]
        x_val = d_test[:, 2:]
        # Compute history representation and one-step ahead prediction with encoder
        g_hat, y_hat_enc, repr = self.forward(x_val, a_val)
        y_hat_enc = y_hat_enc.detach().numpy()
        #Re-Scale predicted outcomes if necessary
        if y_scaler is not None:
            y_hat_enc = y_scaler.inverse_transform(np.expand_dims(y_hat_enc,1))
            y_hat_enc = y_hat_enc[:,0]
        return g_hat.detach().numpy(),y_hat_enc, repr.detach().numpy()

    def estimate_ate(self,d_test,a_int,g_hat=None,y_scaler=None):
        n = d_test.shape[0]
        d_test_int1 = d_test.copy()
        d_test_int1[:, 1] = np.full(n,1)
        _, y_hat_int1, _ = self.test_eval(d_test_int1,y_scaler)
        d_test_int0 = d_test.copy()
        d_test_int0[:, 1] = np.full(n,0)
        _, y_hat_int0, _ = self.test_eval(d_test_int0,y_scaler)

        #Remove if overlapping assumption violated
        if g_hat is not None:
            keep_these = np.logical_and(g_hat >= 0.01, g_hat <= 1. - 0.01)
            y_hat_int1 = y_hat_int1[keep_these]
            y_hat_int0 = y_hat_int0[keep_these]

        # Estimated ATE
        ate_hat = np.mean(y_hat_int1 - y_hat_int0)
        #ate_hat = np.mean(y_hat_int)
        return ate_hat

class DragonNet_Static_Double(DragonNet_Static):
    def __init__(self, input_size, hidden_size, y_size, hidden_size_dragon, dropout):
        super(DragonNet_Static_Double, self).__init__(input_size, hidden_size,dropout)
        #Head layers
        self.layer11 = nn.Linear(hidden_size, hidden_size_dragon)
        self.layer12 = nn.Linear(hidden_size_dragon, hidden_size_dragon)
        self.layer13 = nn.Linear(hidden_size_dragon, y_size)
        self.layer21 = nn.Linear(hidden_size, hidden_size_dragon)
        self.layer22 = nn.Linear(hidden_size_dragon, hidden_size_dragon)
        self.layer23 = nn.Linear(hidden_size_dragon, y_size)

    def forward(self, x,treat_ind):
        # x is of size (batch_size,input_size)
        # treat_ind is of size n
        n = x.size(0)
        repr = fctnl.elu(self.body1(x))
        repr = fctnl.elu(self.body2(repr))
        repr = fctnl.elu(self.body3(repr))
        g_hat = torch.squeeze(torch.sigmoid(self.propnet(repr)))
        #Head 1
        y_1_hat = fctnl.elu(self.layer11(repr))
        y_1_hat = fctnl.elu(self.layer12(y_1_hat))
        y_1_hat = torch.squeeze(self.layer13(y_1_hat))
        # head 2
        y_0_hat = fctnl.elu(self.layer21(repr))
        y_0_hat = fctnl.elu(self.layer22(y_0_hat))
        y_0_hat = torch.squeeze(self.layer23(y_0_hat))
        #Output
        y_hat = treat_ind * y_1_hat + (torch.ones(n) - treat_ind) * y_0_hat
        return g_hat, y_hat, repr


class DragonNet_Static_Single(DragonNet_Static):
    def __init__(self, input_size, hidden_size, y_size, hidden_size_dragon, dropout):
        super(DragonNet_Static_Single, self).__init__(input_size, hidden_size, dropout)
        #Single Head
        self.layer1 = nn.Linear(hidden_size + 1, hidden_size_dragon)
        self.drop_layer1 = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_size_dragon, y_size)

    def forward(self, x,treat_ind):
        # x is of size (batch_size,input_size)
        # treat_ind is of size n
        repr = fctnl.relu(self.drop_body1(self.body1(x)))
        repr = fctnl.elu(self.drop_body2(self.body2(repr)))
        g_hat = torch.squeeze(torch.sigmoid(self.propnet(repr)))
        #Add Treatment to representation
        treat_ind = torch.unsqueeze(treat_ind, 1)
        drag_input = torch.cat((repr, treat_ind), 1)
        #Head 1
        y_hat = fctnl.relu(self.drop_layer1(self.layer1(drag_input)))
        y_hat = torch.squeeze(self.layer2(y_hat))
        return g_hat, y_hat, repr


class Suff_Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        #Head layers
        self.layer1 = nn.Linear(input_size, 1)

    def forward(self,x):
        out = self.layer1(x)
        return out[:,0]

    def train_procedure(self,d_train, batch_size, lr,n_epochs=100,moment=0):
        data = torch.from_numpy(d_train.astype(np.float32))
        train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)
        optimizer_enc = torch.optim.SGD(self.parameters(), lr=lr,momentum=moment,nesterov=False)
        #optimizer_enc = torch.optim.Adam(self.parameters(), lr=lr)

        def obj(y_hat, y):
            outcome_loss = torch.mean((y_hat - y) ** 2)
            return outcome_loss
        # Train
        print('train encoder')
        for epoch in range(n_epochs):
            for batch in train_loader:
                # Process data for encoder
                y_data = batch[:, 0]
                input_data = batch[:, 1:]
                # Forward pass
                y_est = self.forward(input_data)
                # Loss
                loss_enc = obj(y_est, y_data)
                # Update
                loss_enc.backward()
                optimizer_enc.step()
                optimizer_enc.zero_grad()
            # if (epoch + 1) % n_epochs == 0:
            #print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss_enc.item():.4f}')

    def test_eval(self, d_test):
        # Data processing for encoder input
        d_test = torch.from_numpy(d_test.astype(np.float32))
        input_data = d_test[:, 1:]
        # Compute history representation and one-step ahead prediction with encoder
        y_hat = self.forward(input_data)
        return y_hat.detach().numpy()

    def estimate_ate(self,d_test,a_int):
        n = d_test.shape[0]
        d_test_int = d_test.copy()
        d_test_int[:, 1] = np.full(n,1)
        y_hat_int = self.test_eval(d_test_int)
        d_test_int0 = d_test.copy()
        d_test_int0[:, 1] = np.full(n,0)
        y_hat_int0 = self.test_eval(d_test_int0)
        # Estimated ATE
        ate_hat = np.mean(y_hat_int - y_hat_int0)
        return ate_hat
