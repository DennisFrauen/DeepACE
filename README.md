DeepACE
==============================
![image](https://github.com/DennisFrauen/DeepACE/blob/main/Doc/DeepACE_architecture.png)

#### Introduction
This repository contains the code to our paper [Estimating average causal effects from patient trajectories](https://github.com/DennisFrauen/DeepACE/blob/main/Doc/DeepACE_submission.pdf).

#### Requirements
The project is build with python 3.9.7 and uses the following packages:
1. [Pytorch 1.10.0, Pytorch lightning 1.5.1] - deep learning models
2. [Optuna 2.10.0] - hyperparameter tuning
3. [EconML 0.12.0] - Static baselines
4. Other: Pandas 1.3.4, numpy 1.21.5, scikit-learn 1.0.1

Some baseline models are implemented as R scripts. For running the models, all packages that are imported at the beginning of the script (via "library()") need to be installed.

#### Datasets


#### Reproducing the experiments
The scripts running the experiments are contained in the /experiments folder. There are three python scripts, one for each dataset (synthetic = sim, semi-synthetic = mimic and real-world = backpain). 
For the synthetic and semi-synthetic experiments, one needs to specify a configuration file in the main running procedure before running the script. This indicates the models used to obtain results. of the respective script, which specifies the models used to obtain results. The following configurations are possible:

1. config_deepace: DeepACE without targeting
2. config_deepace_tar: DeepACE with targeting
3. config_ltmle_super: LTMLE with super learner
4. config_other: other longitudinal baselines
5. config_gnet: G-Net
6. config_static: Static baselines

The corresponindg .yaml configuration files can be found in /experiments/conifg/. Here, the "treat" parameter denotes the treatment configuration (setting) and takes values in {1,2,3}.

#### Reproducing hyperparameter tuning
