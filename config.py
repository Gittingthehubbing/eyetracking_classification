import numpy as np

config_dict = {
    "plot_folder_name": "plots",
    "data_save_folder_name": "saved_results",
    "plot_some_examples": False,
    "plot_histograms": True,
    "plot_learning_curves":True,
    "save_learning_curves":True,
    "max_seq_length": 20,
    "every_nth_entry": 20,
    "use_padding": True,
    "normalise_x_y_within_sample": False,
    "one_hot_y": False,
    "batch_size": 64,
    "num_epochs": 200,
    "use_reduced_set": False ,# reduces number of samples used
    "model_to_use": ["BERT","LSTM"][0],
    "n_layers_BERT": 8, 
    "head_multiplication_factor": 16,
    "num_attention_heads": 2,
    "hidden_dim_lstm":256,
    "n_layers_LSTM":1,
    "lr": 8e-5,
    "lr_LSTM": 1e-04,
    "lr_scheduling": ["const","ExponentialLR","multistep","anneal"][1],
    "lr_sched_exp_fac": 0.99,
    "multistep_milestones": [25,50,75,100],
    "gamma_multistep": 0.5,
    "min_lr_anneal": 1e-6,
    "redo_samples": True, #if loop requires changes to dataset
    "loop_var_name": "__",
    "loop_vals" : [1],#np.arange(25,200,25),
    "loop_plot_xscale":["linear","log"][0]
}

config_dict["batch_size"] = config_dict["batch_size"] if config_dict["use_padding"] else 1
config_dict["last_activation"] = "Softmax" if config_dict["one_hot_y"] else "Sigmoid"


if __name__ == "__main__":
    import main
    main.main(config_dict)