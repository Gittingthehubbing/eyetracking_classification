import numpy as np
from matplotlib import pyplot as plt
import pathlib as pl
import torch as t
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

def prep_single_sample(f_path, every_nth_entry:int, max_seq_length:int, normalise_x_y_within_sample:bool):
    """Loads file into array and preprocesses it."""


    if not f_path.is_file():
        print(f"{f_path} missing")
    exp_arr = np.genfromtxt(f_path)

    if exp_arr.shape[0] <= 2* every_nth_entry:
        return None
    
    exp_arr[:,0] = exp_arr[:,0]-exp_arr[:,0].min()
    if normalise_x_y_within_sample:
        exp_arr[:,1] = exp_arr[:,1]-exp_arr[:,1].min()
        exp_arr[:,1] = exp_arr[:,1]/np.max(exp_arr[:,1])
        
        exp_arr[:,2] = exp_arr[:,2]-exp_arr[:,2].min()
        exp_arr[:,2] = exp_arr[:,2]/np.max(exp_arr[:,2])

    if exp_arr.shape[0] != exp_arr[:,0].max()+1:
        print(f"{f_path} has missing datapoints")

    exp_arr = exp_arr[0::every_nth_entry,:] # makes array sparse by skipping timesteps

    if exp_arr.shape[0]-1 >= max_seq_length:
        exp_arr = exp_arr[-max_seq_length-1:-1,:] #reduces number of timesteps used
    return exp_arr

def plot_sample_data(data_list, plot_path, one_hot_y):
    """Plots samples in grid of plots."""

    data_indeces = [np.random.randint(0,len(data_list)) for _ in range(2*4)]
    fig,axs = plt.subplots(2,2,sharex=False,sharey=False, figsize=(4,4),dpi=200)
    axs = axs.ravel()
    for idx_plot, ax in enumerate(axs):
        idx_plot = data_indeces[idx_plot]
        data_plot = data_list[idx_plot][0]
        x_plot = data_plot[:,0]
        y_plot = data_plot[:,1]
        result_plot = data_list[idx_plot][1][0] if one_hot_y else data_list[idx_plot][1]
        color_plot = "blue" if result_plot == 1 else "red"
        ax.plot(x_plot,y_plot,'.-',color = color_plot)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Target = {result_plot}")
    plt.tight_layout()
    plt.savefig(plot_path.joinpath("Example_data_noShare.png"),dpi=200)
    plt.close("all")

def plot_histograms(x_arrs_full, col_names, plot_path, target_arr):
    """Plots histograms of input and target data."""

    fig,axs = plt.subplots(1,3,sharex=False,sharey=True, figsize=(8,4),dpi=200)
    axs = axs.ravel()
    for ax_idx, ax in enumerate(axs):
        ax.hist(x_arrs_full[:,ax_idx],bins=50)
        ax.set_title(f"{col_names[ax_idx+1]}")
        ax.set_xlabel(f"Values for {col_names[ax_idx+1]}")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(plot_path.joinpath("Input_data_hist.png"),dpi=200)
    plt.close("all")

    plt.bar("0",len(target_arr[target_arr==0]),label="0",width=0.5)
    plt.bar("1",len(target_arr[target_arr==1]),label="1",width=0.5)
    plt.xlabel("Values for Target")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot_path.joinpath("Target_data_hist.png"),dpi=200)
    plt.close("all")


def evaluate_predictions(predictions_numpy, target_vals_numpy):
    
    accuracy_score_final = accuracy_score(predictions_numpy, target_vals_numpy)
    recall  = recall_score(predictions_numpy, target_vals_numpy)
    f1_score_final  = f1_score(predictions_numpy, target_vals_numpy)
    precision_score_final  = precision_score(predictions_numpy, target_vals_numpy)

    print(f"Final accuracy_score_final for model is {accuracy_score_final*100:.2f}%")
    print(f"Final recall for model is {recall*100:.2f}%")
    print(f"Final precision_score_final for model is {precision_score_final*100:.2f}%")
    print(f"Final f1_score_final for model is {f1_score_final*100:.2f}%")

    return accuracy_score_final