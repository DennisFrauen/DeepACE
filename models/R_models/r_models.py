import numpy
import numpy as np
from pathlib import Path
import os
import subprocess
import sklearn as sk

def save_to_csv(data, path = None):
    n = data.shape[0]
    T = data.shape[1]
    p = data.shape[2]
    data_2d = np.reshape(data.copy().transpose(0,2,1),(n*p,T))
    np.savetxt(path+'/data_seq.csv', data_2d, delimiter=",")


def run_r_model(d_train_unscaled, p, method, a_int1, a_int2, Rpath ="C:/Users/frauend/Documents/R/R-4.1.2/bin/x64/Rscript.exe"):
    #Scale covariates
    n = d_train_unscaled.shape[0]
    T = d_train_unscaled.shape[1]
    x_data = d_train_unscaled[:, :, 2:]
    for i in range(x_data.shape[2]):
        vec_unscaled = np.reshape(x_data.copy()[:,:,i],(n*T,1))
        scaler = sk.preprocessing.StandardScaler().fit(vec_unscaled)
        vec_scaled = scaler.transform(vec_unscaled)[:, 0:1]
        x_data[:, :, i] = np.reshape(vec_scaled.copy(),(n,T))
    d_train_scaled = d_train_unscaled.copy()
    d_train_scaled[:, :, 2:] = x_data
    # Paths
    #Rpath = 'C:/Program Files/R/R-4.1.2/bin/Rscript.exe'    #Specify R path here
    path = Path(os.path.dirname(os.path.realpath(__file__)))

    #Choose R script to run
    if method in ["ltmle", "gcomp", "ltmle_super", "working_msm"]:
        script_path = str(path) + "/ltmle.R"
    if method == "msm":
        script_path = str(path) + "/msm.R"
    if method == "snm":
        script_path = str(path) + "/snm.R"
    if method == "gcomp_par":
        script_path = str(path) + "/gcomp_par.R"
    #Write data to csv
    save_to_csv(d_train_scaled, path=str(path))
    #Arguments for R script
    a1 = list(map(str, a_int1.tolist()))
    a2 = list(map(str, a_int2.tolist()))
    args = [str(p), method] + a1 + a2
    #Command
    cmd = [Rpath, script_path] + args
    #Run tmle script
    try:
        result = subprocess.check_output(cmd, universal_newlines=True)
    except:
        print(f"Error in {method}")
        result = np.nan
    if method == "snm" and np.max(a_int2) > 0:
        print("SNM can't calculate ACE")
        return numpy.NAN
    else:
        return float(result)