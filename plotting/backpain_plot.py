import joblib
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import models.helpers as helpers
import numpy as np

def plt_different_methods():
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    save_dir = str(path.parent.absolute()) + "/results/"
    df_backpain = joblib.load(save_dir + "results_backpain_deepace.pkl")

    methods = df_backpain.columns
    method_names = []
    for i in range(len(methods)):
        if methods[i] == "deepace_tar":
            method_names.append("DeepACE")
        if methods[i] == "gcomp":
            method_names.append("Iter. G-computation")
        if methods[i] == "gcomp_par":
            method_names.append("Par. G-computation")
        if methods[i] == "ltmle":
            method_names.append("LTMLE")
    groups = np.arange(df_backpain.shape[0])
    width = 0.20

    bars = []
    pos = []
    for i, method in enumerate(methods):
        bars.append(df_backpain[method])
        if i == 0:
            pos.append(np.arange(len(bars[0])))
        else:
            pos.append([x + width for x in pos[i - 1]])

    colors = ["red", "blue", "black", "grey"]
    for i, method in enumerate(methods):
        plt.bar(pos[i], bars[i], color=colors[i], edgecolor='white', width=width, label=method_names[i])

    groupnames = []
    for i in groups:
        groupnames.append("Group " + str(i))
    # Add xticks on the middle of the group bars
    plt.xlabel('Patient subgroup')
    plt.xticks([r + width for r in range(len(bars[0]))], groupnames)
    plt.ylabel("ACE estimation error")
    # Create legend & Show graphic
    plt.legend(loc="upper left")
    plt.savefig(helpers.get_project_path() + "/plotting/backpain_bar2.pdf")
    plt.show()

def plt_deepace():
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    save_dir = str(path.parent.absolute()) + "/results/"
    df_deepace = joblib.load(save_dir + "results_backpain_deepace.pkl")
    df_group1 = pd.DataFrame(columns=["Estimated ACE", "Group"])
    df_group1.iloc[:, 0] = df_deepace.iloc[:, 0]
    df_group1.iloc[:, 1] = np.ones(20)
    df_group2 = pd.DataFrame(columns=["Estimated ACE", "Group"])
    df_group2.iloc[:, 0] = df_deepace.iloc[:, 1].to_numpy()
    df_group2.iloc[:, 1] = 2*np.ones(20)
    #Remove outliers
    df_group2 = df_group2[(df_group2["Estimated ACE"] < 0.8)]
    df_plot = pd.concat([df_group1.reset_index(drop=True), df_group2], axis=0)
    df_plot["Estimated ACE"] = df_plot["Estimated ACE"].astype(float)
    df_plot["Group"] = df_plot["Group"].astype(int)

    #Plot
    sns.set_theme(style="whitegrid")
    sns.violinplot(x="Group", y="Estimated ACE", data=df_plot)
    plt.savefig(helpers.get_project_path() + "/plotting/backpain_deepace.pdf")
    plt.show()

if __name__ == "__main__":
    plt_deepace()


