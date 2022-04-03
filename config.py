import numpy as np

config_dict = {
    "plot_folder_name": "plots",
    "data_save_folder_name": "saved_results",
    "plot_some_examples": True,
    "plot_histograms": True,
    "max_seq_length": 600,#50 # 300 good for BERT # max is 1582 min is 23
    "every_nth_entry": 1 ,#28 works well for BERT with max 50 seq. 40 # up to 40 seems to work for BERT
    "use_padding": True,
    "normalise_x_y_within_sample": False,
    "one_hot_y": False,
    "batch_size": 64,
    "num_epochs": 125,
    "use_reduced_set": False ,# reduces number of samples used
    "model_to_use": ["BERT","LSTM"][1],
    "n_layers": 2,
    "head_multiplication_factor": 64,
    "num_attention_heads": 8,
    "hidden_dim_lstm":256,
    "lr": 2.5e-5,
    "lr_LSTM": 1.83298071e-05,
    "lr_scheduling": ["const","ExponentialLR","multistep","anneal"][1],
    "lr_sched_exp_fac": 0.99,
    "multistep_milestones": [25,50,75,100],
    "gamma_multistep": 0.5,
    "min_lr_anneal": 1e-6,
    "redo_samples": True, #if loop requires chages to dataset
    "loop_var_name": "every_nth_entry",
    "loop_vals" : [1,*range(5,45,5)],
}

config_dict["batch_size"] = config_dict["batch_size"] if config_dict["use_padding"] else 1
config_dict["last_activation"] = "Softmax" if config_dict["one_hot_y"] else "Sigmoid"


if __name__ == "__main__":
    import main
    main.main(config_dict)