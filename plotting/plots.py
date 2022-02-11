import math

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.pyplot import cm

#Plot TSNE of representation (to check for treatment predictiveness)
def plot_TSNE(repr, treat,title = None):
    #Repr (n,p), treat (n)
    tsne = TSNE(n_components=2, n_iter=300)
    embedd = tsne.fit_transform(repr)
    df_plot = pd.DataFrame(columns=['x','y','treat'],index=list(range(repr.shape[0])))
    df_plot.iloc[:,0:2] = embedd
    df_plot.iloc[:,2] = treat
    groups = df_plot.groupby('treat')
    for name, group in groups:
        plt.plot(group.x, group.y, marker='o', linestyle='', markersize=5, label=name)
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()


#Plot outcome trajectory together with given treatment indicators
def plot_trajectory(data, data_cf, y_hat1 = None, y_hat2 = None, y_hat3 = None,y_hat4 = None, title = None):
    #data and data_cf are of size (seq_len, p+2), y_hat of size seq_len
    T = data.shape[0]
    x = range(0,T)
    y_fact = data[:,0]
    y_cf = data_cf[:, 0]
    a_fact = data[:, 1]
    a_cf = data_cf[:, 1]
    fig, ax = plt.subplots(2)
    ax[0].plot(x,y_fact, color='green', label='Fact. Outcomes')
    ax[0].plot(x,y_cf, color='red', label='CF. Outcomes')
    if y_hat1 is not None:
        ax[0].plot(x, y_hat1, color='blue', label='alpha = 0')
    if y_hat2 is not None:
        ax[0].plot(x, y_hat2, color='black', label='alpha = 0.5')
    if y_hat3 is not None:
        ax[0].plot(x, y_hat3, color='yellow', label='alpha = 2')
    if y_hat4 is not None:
        ax[0].plot(x, y_hat4, color='grey', label='alpha = 20')
    ax[1].step(x,a_fact, color='green', label='Fact. Treatments')
    ax[1].step(x,a_cf, color='red', label='Interventions')
    ax[0].legend(loc='upper left')
    if title is not None:
        plt.title(title)
    plt.show()

#General plotting procedure for longitudinal trajectories (e.g. outcomes, treatments)
def plot_trajectory_general(traj1,label1,traj2,label2,traj3=None,label3=None,traj4=None,label4=None,title=None):
    x = range(0, traj1.shape[0])
    plt.plot(x, traj1, color='blue', label=label1)
    plt.plot(x, traj2, color='red', label=label2)
    if traj3 is not None:
        plt.plot(x, traj3, color='green', label=label3)
    if traj4 is not None:
        plt.plot(x, traj4, color='yellow', label=label4)
    plt.legend(loc='upper left')
    if title is not None:
        plt.title(title)
    plt.show()

def plot_graph(seq,label,title=None):
    x = range(0, seq.shape[0])
    plt.plot(x, seq, color='blue', label=label)
    if title is not None:
        plt.title(title)
    plt.show()


#Plot longitudinal data (factual and cf) trajectories
def plot_data(d_seq_unscaled, d_seq_cf_unscaled1, d_seq_cf_unscaled2, observation_id=0, cov_id = 0):
    #Plot data
    # Outcome trajectories
    y_true = d_seq_unscaled[observation_id, :, 0]
    y_cf1 = d_seq_cf_unscaled1[observation_id, :, 0]
    y_cf2 = d_seq_cf_unscaled2[observation_id, :, 0]
    plot_trajectory_general(y_true, 'Fact Outcomes1', y_cf1, 'CF Outcomes1', y_cf2, 'CF Outcomes2')
    # Covariates
    x_train = d_seq_unscaled[observation_id, :, cov_id+2]
    x_train_cf1 = d_seq_cf_unscaled1[observation_id, :, cov_id+2]
    x_train_cf2 = d_seq_cf_unscaled2[observation_id, :, cov_id+2]
    plot_trajectory_general(x_train, 'Fact Cov', x_train_cf1, 'CF Cov 1', x_train_cf2, 'CF Cov 2')
    # Treatment
    a_true = d_seq_unscaled[observation_id, :, 1]
    a_cf1 = d_seq_cf_unscaled1[observation_id, :, 1]
    a_cf2 = d_seq_cf_unscaled2[observation_id, :, 1]
    plot_trajectory_general(a_true, 'Fact treatment', a_cf1, 'CF treatment 1', a_cf2, 'CF treatment 2')


#Input: Dataframe with column being different method errors, nrows = tau
def plot_results(results):
    #Colors
    colors = cm.rainbow(np.linspace(0, 1, len(results.columns)))
    x = range(0, results.shape[0])
    for i, col in enumerate(results.columns):
        plt.plot(x, np.log(results[col]), color= colors[i], label=col)
    plt.legend(loc='upper left')
    plt.title("Encoder results")
    plt.xlabel("ATE forecast horizon")
    plt.ylabel("Average estimation error (log)")
    plt.savefig('results.png')

