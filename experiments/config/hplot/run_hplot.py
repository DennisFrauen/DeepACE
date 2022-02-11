import numpy as np

import models.helpers as helpers
import joblib
import yaml
import experiments.run_experiments as mainrun
import pandas as pd

def load_hyperparam(config, h):
    methods = config["methods"]
    method_params = {}
    save_dir_hyper = helpers.get_project_path() + "/hyperparam/parameters/hplot/"
    for method in methods:
        if method=="deepace_tar":
            save_dir_hyper_d = save_dir_hyper + "deepace_tar/study_deepace_tar_h" + str(h) + "_int"
        elif method=="deepace":
            save_dir_hyper_d = save_dir_hyper + "deepace/study_deepace_tar_h" + str(h) + "_int"
        if method in ["deepace_tar", "deepace"]:
            params1 = joblib.load(save_dir_hyper_d + "1.pkl").best_trial.params
            params2 = joblib.load(save_dir_hyper_d + "2.pkl").best_trial.params
            method_params[method] = [params1, params2]
        else:
            method_params[method] = None
    return method_params


if __name__ == "__main__":
    # Coniguration for run
    run_config_name = "config_hplot_deepace"
    stream = open(helpers.get_project_path() + "/experiments/config/hplot/" + run_config_name + ".yaml", 'r')
    config = yaml.safe_load(stream)
    df_hplot = pd.DataFrame(columns=config["methods"] + [m + "_var" for m in config["methods"]], index=np.arange(8))
    n_methods = len(config["methods"])
    for h in range(2, 7):
        print(f"Lag h={h}")
        config["lag"] = h
        #Load hyperparameters for lag h
        method_params = load_hyperparam(config, h)
        df_results = mainrun.perform_experiments(run_config=config, load_hyper=False, method_params=method_params, return_results=True)
        df_hplot.iloc[h - 1, 0:n_methods] = df_results.iloc[:, 1].to_numpy()
        df_hplot.iloc[h - 1, n_methods:] = df_results.iloc[:, 2].to_numpy()
    print(df_hplot)
    if run_config_name == "config_hplot_deepace":
        method_name = "deepace"
    if run_config_name == "config_hplot_deepace_tar":
        method_name = "deepace_tar"
    if run_config_name == "config_hplot_gcomp":
        method_name = "gcomp"
    if run_config_name == "config_hplot_ltmle":
        method_name = "ltmle"
    #Save results
    joblib.dump(df_hplot, helpers.get_project_path() + "/results/results_hplot_" + method_name + ".pkl")