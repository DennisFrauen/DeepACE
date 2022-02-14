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


#### Reproducing experiments on synthetic data

#### Reproducing experiments on semi-synthetic data

#### Reproducing experiments on real-word data

#### Reproducing hyperparameter tuning
