import numpy as np

config_dict = {
    "plot_folder_name": "plots",
    "data_save_folder_name": "saved_results",
    "plot_some_examples": False,
    "plot_histograms": True,
    "plot_learning_curves":True,
    "save_learning_curves":True,
    "max_seq_length": 600,
    "every_nth_entry": 20,#28 works well for BERT with max 50 seq. 40 # up to 40 seems to work for BERT
    "use_padding": True,
    "normalise_x_y_within_sample": False,
    "one_hot_y": False,
    "batch_size": 64,
    "num_epochs": 125,
    "use_reduced_set": False ,# reduces number of samples used
    "model_to_use": ["BERT","LSTM"][1],
    "n_layers_BERT": 8, #2 for BERT
    "head_multiplication_factor": 16,
    "num_attention_heads": 2,
    "hidden_dim_lstm":256,
    "n_layers_LSTM":1,
    "lr": 2.5e-5,
    "lr_LSTM": 1.832e-05,
    "lr_scheduling": ["const","ExponentialLR","multistep","anneal"][1],
    "lr_sched_exp_fac": 0.99,
    "multistep_milestones": [25,50,75,100],
    "gamma_multistep": 0.5,
    "min_lr_anneal": 1e-6,
    "redo_samples": True, #if loop requires chages to dataset
    "loop_var_name": "max_seq_length",
    "loop_vals" : np.arange(50,600,50),
    "loop_plot_xscale":["linear","log"][0]
}

config_dict["batch_size"] = config_dict["batch_size"] if config_dict["use_padding"] else 1
config_dict["last_activation"] = "Softmax" if config_dict["one_hot_y"] else "Sigmoid"


if __name__ == "__main__":
    import main
    main.main(config_dict)