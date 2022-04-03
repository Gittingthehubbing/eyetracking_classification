"""
Title: Eye-tracking binary classification
Author: Thomas Mercier
Date created: 2022/03/29
Description: Loads, preprocesses eye-tracking data and implements different ML models to make predictions.
"""
from torch.utils.data import Dataset as torch_dset


class DSet(torch_dset):

    def __init__(self, x_list,y_list) -> None:
        super().__init__()
        self.x_list =x_list
        self.y_list =y_list

    def __getitem__(self, index):
        return self.x_list[index], self.y_list[index]

    def __len__(self):
        return len(self.x_list)

def main(cfg:dict):
    """Loads in data, trains and evaluates model according to settings in cfg dict."""
        
    import pandas as pd
    import torch as t
    from torch import nn
    from torch.utils.data import Subset as torch_subset
    from torch.utils.data.dataloader import DataLoader as dl
    import pathlib as pl
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    import seaborn as sb
    
    import models
    import ml_utils

    plot_path = pl.Path(cfg["plot_folder_name"])
    plot_path.mkdir(exist_ok=True,parents=False)

    txt_save_path = pl.Path(cfg["data_save_folder_name"])
    txt_save_path.mkdir(exist_ok=True,parents=False)

    loop_losses_train = []
    loop_losses_val = []

    data_path = pl.Path("eyetracker_data")
    raw_data_files = list(data_path.joinpath("data").glob("N*.txt"))
    target_arr = np.genfromtxt(data_path.joinpath("Outcome.txt"))

    data_indeces_full =np.arange(target_arr.shape[0])
    np.random.seed(0)
    np.random.shuffle(data_indeces_full)

    train_indeces_full = data_indeces_full[:int(0.8*len(data_indeces_full))] #used later to make sure same samples are used
    val_indeces_full =  data_indeces_full[int(0.8*len(data_indeces_full)):]

    device_torch = "cuda" if t.cuda.is_available() else "cpu"

    col_names = ["times", "x", "y", "pupil_size", "dropme"]

    batch_size_original = cfg["batch_size"]

    for idx_loop, loop_val in enumerate(cfg["loop_vals"]):
        cfg[cfg["loop_var_name"]] = loop_val
        if idx_loop == 0 or cfg["redo_samples"]:               

            if cfg["one_hot_y"]:
                target_arr_before = target_arr.copy()
                target_arr_one_hot = np.zeros((target_arr.shape[0],2))
                for y_val_idx, y_val in enumerate(target_arr):
                    target_arr_one_hot[y_val_idx,:] = [1 if y_val == 0 else 0 ,1 if y_val == 1 else 0]
                target_arr = target_arr_one_hot
            elif len(target_arr.shape) > 1:
                target_arr = target_arr_before

            cfg["batch_size"] = batch_size_original if cfg["use_padding"] else 1
            data_list = []
            sequence_lenghts = []
            for idx in tqdm(range(1,len(raw_data_files)+1),desc="Preprocessing samples"):
                is_train_sample = True if idx in train_indeces_full else False            
                f_path = raw_data_files[0].parent.joinpath(f"N{idx}.txt")

                exp_arr = ml_utils.prep_single_sample(f_path, cfg["every_nth_entry"], cfg["max_seq_length"], cfg["normalise_x_y_within_sample"])
                if exp_arr is None:
                    continue
                sequence_lenghts.append(exp_arr.shape[0])
                data_list.append((exp_arr[:,1:-1], target_arr[idx-1])) # last column  is nan

                if idx ==1:
                    x_arrs_full = exp_arr[:,1:-1]
                    if is_train_sample:
                        x_arrs_for_scaling = exp_arr[:,1:-1]
                else:
                    x_arrs_full = np.concatenate([x_arrs_full,exp_arr[:,1:-1]],axis=0)
                    if is_train_sample:
                        x_arrs_for_scaling = np.concatenate([x_arrs_for_scaling,exp_arr[:,1:-1]],axis=0)
            max_seq_length_found = np.max(sequence_lenghts)
            print(f"Max sequence length used is: {max_seq_length_found}")

            if cfg["plot_histograms"]:
                plt.hist(sequence_lenghts,bins=100)
                plt.xlabel("Sequence Lengths")
                plt.savefig("Seq_lengths.png",dpi=200)
                plt.close("all")

            if cfg["plot_some_examples"]:
                ml_utils.plot_sample_data(data_list, plot_path, cfg["one_hot_y"])

            if cfg["plot_histograms"]:
                ml_utils.plot_histograms(x_arrs_full, col_names, plot_path, target_arr)

            x_scaler = StandardScaler()
            x_scaler.fit(x_arrs_for_scaling)
            
            x_list, y_list = [], []
            for k in data_list:
                x_val = t.tensor(x_scaler.transform(k[0]),dtype=t.float32)                
                x_list.append(x_val)
                y_list.append(t.tensor(k[1],dtype=t.float32))

            if cfg["use_padding"]:
                x_list_padded = []
                for x_val in x_list:
                    sequence_length_diff = max_seq_length_found - x_val.shape[0]
                    if sequence_length_diff > 0:
                        x_pad = t.zeros((sequence_length_diff, x_val.shape[1]),dtype=x_val.dtype, device= x_val.device)
                        x_val_padded = t.cat([x_pad, x_val],dim=0)
                        x_list_padded.append(x_val_padded)
                    else:
                        x_list_padded.append(x_val)
                x_vals = t.stack(x_list_padded,dim=0)
                y_vals = t.stack(y_list,dim=0)
            else:
                x_vals = x_list
                y_vals = y_list

            data_indeces = np.arange(len(x_vals))
            train_indeces = [x for x in data_indeces if x in train_indeces_full]
            val_indeces =  [x for x in data_indeces if x in val_indeces_full]
            print(f"Total number of positive samples: {len(target_arr[target_arr==0])}")
            print(f"Total number of training samples: {len(train_indeces)}")
            print(f"Total number of validation samples: {len(val_indeces)}")

            full_set = DSet(x_vals,y_vals)

            train_set = torch_subset(full_set,train_indeces)
            val_set = torch_subset(full_set,val_indeces)

        num_workers = 0
        if cfg["use_reduced_set"]:        
            train_set_reduced = torch_subset(train_set,[x for x in range(int(len(train_set)/4))])
            val_set_reduced = torch_subset(val_set,[x for x in range(int(len(val_set)/4))])
            train_loader = dl(train_set_reduced,cfg["batch_size"],shuffle=True, num_workers=num_workers)
            val_loader = dl(val_set_reduced,cfg["batch_size"],shuffle=False, num_workers=num_workers)
        else:
            train_loader = dl(train_set,cfg["batch_size"],shuffle=True, num_workers=num_workers)
            val_loader = dl(val_set,cfg["batch_size"],shuffle=False, num_workers=num_workers)

        for x,y in train_loader:
            x0 = x.to(device_torch)
            y0 = y.to(device_torch)
            break
        hidden_dim_bert = cfg["head_multiplication_factor"] * cfg["num_attention_heads"]

        if cfg["model_to_use"] == "BERT":
            net = models.BERT_Model(
                x0,y0,cfg["num_attention_heads"],hidden_dim_bert,cfg["n_layers_BERT"],
                last_activation=cfg["last_activation"],max_seq_length=max_seq_length_found
            ).to(device=device_torch)
        elif cfg["model_to_use"] == "LSTM":
            net = models.LSTM_Model(
                x0,y0,cfg["hidden_dim_lstm"],cfg["n_layers_LSTM"],
                zero_initial_h_c=True,last_activation=cfg["last_activation"]
            ).to(device_torch)
        else:
            raise NotImplementedError("Model not implemented")

        numParams = sum([i.numel() for i in net.parameters()])
        print(f"Model has {(numParams/1e3):.2f}k parameters")

        crit = nn.BCELoss()
        learning_rate = cfg["lr_LSTM"] if cfg["model_to_use"] == "LSTM" else cfg["lr"]
        opti = t.optim.Adam(net.parameters(),lr=learning_rate)

        
        if cfg["lr_scheduling"] == "multistep":
            scheduler = t.optim.lr_scheduler.MultiStepLR(
                opti, milestones=cfg["multistep_milestones"], gamma=cfg["gamma_multistep"], verbose=False)
        elif cfg["lr_scheduling"] == "anneal":
            scheduler = t.optim.lr_scheduler.CosineAnnealingLR(
                opti, 250, eta_min=cfg["min_lr_anneal"], last_epoch=-1, verbose=False)
        elif cfg["lr_scheduling"] == "ExponentialLR":
            scheduler = t.optim.lr_scheduler.ExponentialLR(opti,cfg["lr_sched_exp_fac"])
        else:
            scheduler = None


        losses_fine=[]
        losses_val_fine = []
        losses_epochs = []
        losses_val_epochs = []
        accuracies_train = []
        accuracies_train_epochs = []
        accuracies_val = []
        accuracies_val_epochs = []
        num = 0
        trainSteps =0
        valSteps=0
        pbar = tqdm(range(cfg["num_epochs"]))
        for e in pbar:
            epochLoss = []
            epochLoss_val = []
            
            for idx,(x,y) in enumerate(train_loader):
                num+=1   
                out = net(x.to(device_torch))
                loss_train = crit(out.view(-1),y.to(device_torch).view(-1))
                opti.zero_grad()
                loss_train.backward()
                opti.step()
                trainSteps+=1
                losses_fine.append(loss_train.detach())
                epochLoss.append(loss_train.detach())
                
                bool_correct_train = t.round(out.view(-1)) == y.to(device_torch).view(-1)
                accucaracy_train = bool_correct_train.sum()/len(out.view(-1))
                accuracies_train.append(accucaracy_train.detach()*100)

                if num % (len(train_loader)/2) == 0 and num > 1:
                    net.eval()
                    with t.inference_mode():
                        for x_val, y_val in val_loader:
                            out = net(x_val.to(device_torch))
                            loss_val = crit(out.view(-1),y_val.to(device_torch).view(-1))
                            bool_correct_val = t.round(out.view(-1)) == y_val.to(device_torch).view(-1)
                            accucaracy_val = bool_correct_val.sum()/len(out.view(-1))
                            accuracies_val.append(accucaracy_val.detach()*100)
                            losses_val_fine.append(loss_val.detach())                        
                            
                            valSteps+=1
                            epochLoss_val.append(loss_val.detach())
                    pbarText = f"Epoch {e} Trainloss {t.mean(t.stack(epochLoss)):.3f}  EvalLoss {t.mean(t.stack(epochLoss_val)):.3f} Acc_val {t.mean(t.stack(accuracies_val)):.3f}% TrainSteps {trainSteps} ValSteps {valSteps}"
                    net.train()
            losses_epochs.append(t.mean(t.stack(epochLoss)).detach())
            losses_val_epochs.append(t.mean(t.stack(epochLoss_val)))
            accuracies_val_epochs.append(t.mean(t.stack(accuracies_val)))
            accuracies_train_epochs.append(t.mean(t.stack(accuracies_train)))
            if scheduler is not None:
                scheduler.step()
            pbar.set_description(pbarText)
            

        losses_epochs_numpy = [x.cpu().numpy().item() for x in losses_epochs]
        losses_epochs_val_numpy = [x.cpu().numpy().item() for x in losses_val_epochs]
        accuracies_val_epochs_numpy = [x.cpu().numpy().item() for x in accuracies_val_epochs]
        accuracies_train_epochs_numpy = [x.cpu().numpy().item() for x in accuracies_train_epochs]

        if cfg["plot_learning_curves"]:
            plt.plot(losses_epochs_numpy,".-",label="train_loss")
            plt.plot(losses_epochs_val_numpy,".-",label="val_loss")
            if cfg["loop_plot_xscale"] == "log":
                plt.xscale("log")
            plt.legend()
            plt.title(f'{cfg["loop_var_name"]} = {loop_val}')
            plt.savefig(plot_path.joinpath(f"{cfg['model_to_use']}_losses_{idx_loop}.png"),dpi=200)
            plt.close("all")

        if cfg["save_learning_curves"]:
            pd.DataFrame({
                "epoch":np.arange(len(losses_epochs_numpy)),
                "Training Loss":losses_epochs_numpy,
                "Validation Loss":losses_epochs_val_numpy,
            },index=np.arange(len(losses_epochs_numpy))).to_csv(txt_save_path.joinpath(f"{cfg['model_to_use']}_Learning_Curve.txt"),index=False)

        if cfg["plot_learning_curves"]:
            plt.plot(accuracies_train_epochs_numpy,label="accuracies_train")
            plt.plot(accuracies_val_epochs_numpy,label="accuracies_val")
            plt.legend()
            plt.title(f'{cfg["loop_var_name"]} = {loop_val}')
            plt.savefig(plot_path.joinpath(f"{cfg['model_to_use']}_accuracies_epochs_numpy_{idx_loop}.png"),dpi=200)
            plt.close("all")

        if cfg["save_learning_curves"]:
            pd.DataFrame({
                "epoch":np.arange(len(accuracies_train_epochs_numpy)),
                "Training Accuracy":accuracies_train_epochs_numpy,
                "Validation Accuracy":accuracies_val_epochs_numpy,
            },index=np.arange(len(accuracies_train_epochs_numpy))).to_csv(txt_save_path.joinpath(f"{cfg['model_to_use']}_Learning_Curve_acc.txt"),index=False)

        loop_losses_train.append(losses_epochs_numpy)
        loop_losses_val.append(losses_epochs_val_numpy)

        y_plot_train = [x[-1] for x in loop_losses_train]
        y_plot_val = [x[-1] for x in loop_losses_val]
        plt.plot(cfg["loop_vals"][:len(y_plot_train)],y_plot_train,".-",label="train")
        plt.plot(cfg["loop_vals"][:len(y_plot_train)],y_plot_val,".-",label="val")
        plt.xlabel(cfg["loop_var_name"])
        plt.ylabel("Final Losse Value")
        if cfg["loop_plot_xscale"] == "log":
            plt.xscale("log")
        plt.legend()
        plt.savefig(plot_path.joinpath(f"{cfg['model_to_use']}_loop_finallosses_val_{cfg['loop_var_name']}.png"),dpi=200)
        plt.close()

        target_vals = []
        predictions = []
        net.eval()
        with t.inference_mode():
            for x_val, y_val in val_loader:
                out = net(x_val.to(device_torch)).detach().cpu().numpy()
                prediction = np.round(out)
                predictions.append(prediction.ravel())
                target_vals.append(y_val.cpu().numpy().ravel())
        predictions_numpy = np.concatenate(predictions,axis=0)
        target_vals_numpy = np.concatenate(target_vals,axis=0)
        conf_matrix = confusion_matrix(predictions_numpy, target_vals_numpy)
        print("Confusion martix:")
        print(conf_matrix)

        accuracy_score_final = ml_utils.evaluate_predictions(predictions_numpy, target_vals_numpy)

        sb.heatmap(conf_matrix,annot=True,fmt=".0f")
        plt.xlabel("Prediction")
        plt.ylabel("Ground Truth")
        plt.savefig(plot_path.joinpath(f"{cfg['model_to_use']}_ConfusionMatrix_{idx_loop}.png"),dpi=200)
        plt.close()

        if idx_loop == 0:
            accuracies_full_valset = np.array([accuracy_score_final*100])
        else:
            accuracies_full_valset = np.concatenate([accuracies_full_valset,np.array([accuracy_score_final*100])],axis=0)
        plt.plot(cfg["loop_vals"][:len(accuracies_full_valset)],accuracies_full_valset,".-",label="val")
        plt.xlabel(cfg["loop_var_name"])
        plt.ylabel("Final Accuracy (%)")
        if cfg["loop_plot_xscale"] == "log":
            plt.xscale("log")
        plt.legend()
        plt.savefig(plot_path.joinpath(f"{cfg['model_to_use']}_loop_final_accuracies_{cfg['loop_var_name']}.png"),dpi=200)
        plt.close()

        print("Loop Values:\n",cfg["loop_vals"][:len(y_plot_train)])
        print("Validation accuracies:\n",accuracies_full_valset)
        pd.DataFrame({
            cfg["loop_var_name"]:cfg["loop_vals"][:len(accuracies_full_valset)],
            "Validation Accuracy":accuracies_full_valset
        },index=np.arange(accuracies_full_valset.shape[0])).to_csv(txt_save_path.joinpath(f"{cfg['model_to_use']}_loop_final_accuracies_{cfg['loop_var_name']}.txt"),index=False)

if __name__ == "__main__":
    from config import config_dict as cfg

    main(cfg)