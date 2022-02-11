import plotting.plots as plotting
import joblib
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import models.helpers as helpers
import numpy as np

if __name__ == "__main__":
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    save_dir = str(path.parent.absolute()) + "/results/"
    df_hplot_deepace_tar = joblib.load(save_dir + "results_hplot_deepace_tar.pkl").iloc[1:6,:]
    df_hplot_deepace = joblib.load(save_dir + "results_hplot_deepace.pkl").iloc[1:6,:]
    df_hplot_gcomp = joblib.load(save_dir + "results_hplot_gcomp.pkl").iloc[1:6,:]
    fig, ax = plt.subplots()
    h = df_hplot_gcomp.shape[0]

    epochs = list(range(5))
    #clrs = sns.color_palette("husl", 4)
    with sns.axes_style("darkgrid"):

        meanst = np.array(df_hplot_gcomp["gcomp"], dtype=np.float64)
        sdt = np.array(df_hplot_gcomp["gcomp_var"], dtype=np.float64)
        ax.plot(epochs, meanst, label="Iter. G-comp.", c="orangered", linestyle="dashed")
        #ax.fill_between(epochs, meanst - sdt, meanst + sdt, alpha=0.3, facecolor=clrs[0])

        meanst = np.array(df_hplot_gcomp["ltmle"], dtype=np.float64)
        sdt = np.array(df_hplot_gcomp["ltmle_var"], dtype=np.float64)
        ax.plot(epochs, meanst, label="LTMLE", c="maroon", linestyle="dashed")
        #ax.fill_between(epochs, meanst - sdt, meanst + sdt, alpha=0.3, facecolor=clrs[1])

        meanst = np.array(df_hplot_deepace["deepace"], dtype=np.float64)
        sdt = np.array(df_hplot_deepace["deepace_var"], dtype=np.float64)
        ax.plot(epochs, meanst, label="DeepACE w/o targeting (ours)", c="deepskyblue", linestyle="dashed")

        meanst = np.array(df_hplot_deepace_tar["deepace_tar"], dtype=np.float64)
        sdt = np.array(df_hplot_deepace_tar["deepace_tar_var"], dtype=np.float64)
        ax.plot(epochs, meanst, label="DeepACE (ours)", c="navy")
        #ax.fill_between(epochs, meanst - sdt, meanst + sdt, alpha=0.3, facecolor="red")

        ax.legend(facecolor="white")
    plt.xlabel(r"Lag $h$")
    plt.ylabel("ACE estimation error")
    plt.xticks(range(h), range(2, h+2))
    plt.savefig(helpers.get_project_path() + "/plotting/hplot.pdf")
    plt.show()
    """
    x = range(0, df_hplot.shape[0])
    plt.plot(x, df_hplot["deepace_tar"], color='blue', label="DeepACE")
    plt.plot(x, df_hplot["ltmle"], color='red', label="LTMLE")
    plt.plot(x, df_hplot["gcomp"], color='green', label="Iter. G-computation")
    plt.legend(loc='upper left')
    plt.show()
    """