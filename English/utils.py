import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
import torchaudio.transforms as T
from pathlib import Path


class AudioMNISTDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = pd.read_csv(data_dir)
        self.base_dir = Path(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio_path = self.base_dir.parent.parent / Path(self.file_list["file_dir"][index])
        label = self.file_list["label"][index]

        audio, sample_rate = torchaudio.load(audio_path)
        
        return audio, label


def get_data(data_dir, bs_train, bs_test, device, data_params):


    # Create training, validation, and testing split of the data
    train_set = AudioMNISTDataset(data_dir + "train.csv")
    validation_set = AudioMNISTDataset(data_dir + "validation.csv")
    test_set = AudioMNISTDataset(data_dir + "test.csv")


    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros

        batch = [item.t() for item in batch]

        #enforce all samples of the same length
        input_length = data_params["initial_input_length"]
        batch.append(torch.zeros(input_length,1, device=device)) #original = 48,000; 240,000

        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch[:-1].permute(0, 2, 1) #don't take last value (artificial)


    def collate_fn(batch):

        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform,  label in batch:
            tensors += [waveform]
            targets += [torch.tensor(label)]


        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors).to(device) #move to device
        targets = torch.stack(targets).to(device) #move to device

        return tensors, targets


    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=bs_train,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=bs_test,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader



def evaluation(data_loader, model, batch_size, objective, device, transform, data_params):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for i, (signals, labels) in enumerate(data_loader):

            signals, labels = signals.to(device), labels.to(device) #move to device

            #appy new sampling rate
            signals = transform(signals)
            #apply a normalization [-1, 1]
            signals = normalize(signals)

            batch_size = signals.shape[0]

            #raw amplitude: reshape to (seq_len, batch, n_inp=1)
            signals = signals.reshape(batch_size, 1, data_params["final_signal_length"])
            input_signal = signals.permute(2, 0, 1)

            #get our predictions
            output = model(input_signal)

            #take softmax over predictions
            softmax = torch.nn.Softmax(dim=1)
            output_probs = softmax(output)

            #get the test loss
            test_loss += objective(output_probs, labels).item()

            #predicted class = argmax over the final readout
            pred = output_probs.argmax(dim=1)

            #how many correct?
            correct += pred.eq(labels).sum()


    test_loss /= (i+1)
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item(), test_loss


def write_results(filename,test_accuracy, test_loss, network):

    fieldnames = ["network", "test_accuracy", "test_loss"]

    #first check if file exists; if not write new file!
    if not os.path.exists(filename):
        #write file!
        with open(filename, mode = "w", newline ='') as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            #write the header 
            writer.writeheader()

    #now, read from file and write results
    with open(filename, mode = "a", newline ='') as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Append new data 
            writer.writerow({"network": network, 
                             "test_accuracy": test_accuracy, 
                             "test_loss": test_loss})
            

    return


def write_network_parameters(filename, config):

    #make sure the results directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode = "w", newline ='') as csvfile:
        #write network parameters
        fieldnames = ["network", "network_type","experiment_id", "n_hid", "epochs", "batch", "learning_rate", "dt", 
                        "random_seed"]
    
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #write the header 
        writer.writeheader()
        writer.writerow({"network": config["network"], 
                    "network_type": config["network_type"],
                    "experiment_id": config["experiment_id"],
                    "n_hid": config["n_hid"],
                    "epochs": config["epochs"],
                    "batch": config["batch"],
                    "learning_rate": config["lr"],
                    "dt": config["dt"], 
                    "random_seed": config["random_seed"]})

    return


def write_hyperparameters(filename, config, model):

    with open(filename, mode = "a", newline ='') as csvfile:

        if config["network"] == "coRNN":

            #write network parameters   

            # gamma # 
            writer = csv.writer(csvfile)
            writer.writerow(["gamma"])

            gamma = model.gamma[0,:].tolist()
            writer.writerow(gamma)

            #epsilon #
            writer = csv.writer(csvfile)
            writer.writerow(["epsilon"])

            epsilon = model.epsilon[0,:].tolist()
            writer.writerow(epsilon)

        else:
            raise Exception("Network {} not supported in this release".format(config["network"]))

    return 


def normalize(signals):

    batch_size = signals.shape[0]

    for batch in range(batch_size):

        max_value = signals[batch,:,:].abs().max()
        signal =  signals[batch,:,:]*(1/max_value)
        signals[batch,:,:] = signal

    return signals

def signal_transform(input_signal, n_inp, device):

    frame_length = int(input_signal.size(0)/n_inp)
    batch_size = input_signal.size(1)

    signals = torch.zeros(size=(frame_length,batch_size,n_inp), device=device)

    end = n_inp
    for index, start in enumerate(range(0,input_signal.size(0), n_inp)):
        signals[index, :, start:end] = input_signal[start:end, :, :].T
        end = start

    return signals







