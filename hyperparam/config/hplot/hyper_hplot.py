import models.helpers as helpers
import yaml
import hyperparam.hyperparameter as hyper_main

if __name__ == "__main__":
    stream = open(helpers.get_project_path() + "/hyperparam/config/hplot/config_hyper_deepace_tar.yaml", 'r')
    hyper_config = yaml.safe_load(stream)

    save_dir_hyper = helpers.get_project_path() + "/hyperparam/parameters/hplot/"
    T = hyper_config["T"]
    number_trials = hyper_config["number_trials"]

    for h in range(1, 7):
        hyper_config["lag"] = h
        tune_sampler = hyper_main.set_seeds(hyper_config["seed"])
        d_train_seq, a_int_1, a_int_2 = hyper_main.generate_data(hyper_config)
        #Hyperparameter tuning deepace
        study_name = "study_deepace_tar_h" + str(h) + "_int"
        print(f"Hyperparameter tuning DeepACE + targeting, intervention 1, h={h}")
        hyper_main.tune_deepace(d_train_seq, a_int_1, study_name=study_name + "1", alpha=0.1, beta=0.01,
                     num_samples=number_trials, path=save_dir_hyper, tune_sampler=tune_sampler)
        print(f"Hyperparameter tuning DeepACE + targeting, intervention 2, h={h}")
        hyper_main.tune_deepace(d_train_seq, a_int_2, study_name=study_name + "2", alpha=0.1, beta=0.01,
                     num_samples=number_trials, path=save_dir_hyper, tune_sampler=tune_sampler)
