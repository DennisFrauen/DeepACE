DeepACE
==============================
![image](https://github.com/DennisFrauen/DeepACE/blob/main/Doc/DeepACE_architecture.png)

#### Introduction
This repository contains the code to our paper [Estimating average causal effects from patient trajectories](https://github.com/DennisFrauen/DeepACE/blob/main/Doc/DeepACE_submission.pdf).
Contact: frauen@lmu.de

#### Requirements
The project is build with python 3.9.7 and uses the following packages:
1. [Pytorch 1.10.0, Pytorch lightning 1.5.1] - deep learning models
2. [Optuna 2.10.0] - hyperparameter tuning
3. [EconML 0.12.0] - Static baselines
4. Other: Pandas 1.3.4, numpy 1.21.5, scikit-learn 1.0.1

Some baseline models are implemented as R scripts. For running the models, all packages that are imported at the beginning of the script (via "library()") need to be installed.

#### Datasets
In our paper we used three datasets: Synthetic, semi-synthetic and real-world data. 

###### Synthetic data
The script for synthetic data generation is `datasets/sim.py`. Here, the data is simulated accourding to Sec. 5.1. in the paper.

###### Semi-synthetic data
We use MIMIC-III, which is accessible but must be requested at https://physionet.org/content/mimiciii/1.4/. When MIMIC-III access is granted, the pre-processed data by Wang et. al. (2020) is accessible with instructions in the respective paper. The preprocessed file needs to be added to `datasets/mimic` and should be named `all_hourly_data.h5`. The script `datasets/mimic/mimic.py` extracts covariates and generated synthetic treatments and outcomes.

###### Real-world data
We use the pre-processed data from the clinical study on low back pain patients from Nielsen et al (2017). The data is available in the folder `datasets/backpain/data_preprocessed`.

#### Reproducing the experiments
The scripts running the experiments are contained in the `/experiments` folder. There are three python scripts, one for each dataset (synthetic = sim, semi-synthetic = mimic and real-world = backpain). 
For the synthetic and semi-synthetic experiments, one needs to specify a configuration file in the main running procedure before running the script. This indicates the models used to obtain results. of the respective script, which specifies the models used to obtain results. The following configurations are possible:

1. `config_deepace`: DeepACE without targeting
2. `config_deepace_tar`: DeepACE with targeting
3. `config_ltmle_super`: LTMLE with super learner
4. `config_other`: other longitudinal baselines
5. `config_gnet`: G-Net
6. `config_static`: Static baselines

The corresponindg .yaml configuration files can be found in `/experiments/conifg/`. Here, the "treat" parameter denotes the treatment configuration (setting) and takes values in {1,2,3}.

#### Reproducing hyperparameter tuning
The hyperparameters for the models trained from the /experiments folder are stored under `/hyperparame/parameters`. For reproducing hyperparameter tuning, one needs to run `hyperparam/hyperparameter.py` (synthetic + semi-synthetic data) or hyperparam/hyperparameter_backpain.py (real-world data). Again, the correct configuration files need to be specified, indicating the models and settings.
