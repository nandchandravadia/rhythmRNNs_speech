"""
Usage:
    python3 task.py --experiment_id 1

All experiment settings live in experiments.json.
"""

import argparse
import csv
import os

import torch
import torchaudio
from torch import nn, optim

import utils
from config import load_config, resolve_parameter
from coRNN import coRNN


# directory this script lives in -- used to resolve the config path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="training parameters")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments.json",
        help="Path to the JSON config file containing experiment definitions",
    )
    parser.add_argument(
        "--experiment_id",
        type=int,
        default=1,
        help="Which experiment_id in the config file to run",
    )
    return parser.parse_args()


def build_model(config, n_inp, n_out, device):
    """Construct the coRNN model from a config dict."""
    n_hid = config["n_hid"]

    gamma = resolve_parameter(config, "gamma", n_hid, device)
    epsilon = resolve_parameter(config, "epsilon", n_hid, device)

    if config["network"] != "coRNN":
        raise Exception(
            "Network {} not supported in this release".format(config["network"])
        )

    model = coRNN(
        network_type=config["network_type"],
        n_inp=n_inp,
        n_hid=n_hid,
        n_out=n_out,
        dt=config["dt"],
        gamma=gamma,
        epsilon=epsilon,
    )
    return model.to(device)


def main():
    args = parse_args()

    # resolve the config path relative to this script if it isn't absolute,
    # so the program works regardless of the current working directory
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(SCRIPT_DIR, config_path)

    config = load_config(config_path, args.experiment_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on Device: {}".format(device))
    print(
        "Loaded experiment {} from {}".format(args.experiment_id, config_path)
    )

    torch.manual_seed(config["random_seed"])

    # ----- Input / output dimensions ---------------------------------------
    # The input is the raw waveform amplitude: one value per timestep.
    n_inp = 1
    n_out = 10

    # ----- Build the model -------------------------------------------------
    model = build_model(config, n_inp, n_out, device)

    # ----- Set up results files -------------------------------------------
    network = "{}_{}".format(config["network"], config["network_type"])
    experiment_id = config["experiment_id"]
    filename = "./results/{}/{}_{}.csv".format(network, network, experiment_id)

    utils.write_network_parameters(filename=filename, config=config)
    utils.write_hyperparameters(filename=filename, config=config, model=model)

    # ----- Data ------------------------------------------------------------
    data_params = {
        "initial_input_length": config["initial_input_length"],
        "final_signal_length": config["final_signal_length"],
        "input_length": config["input_length"],
    }

    train_loader, test_loader = utils.get_data(
        config["data_dir"],
        config["batch"],
        config["batch"],
        device,
        data_params,
    )

    # ----- Signal transform -----------------------------------------------
    # The raw waveform is resampled, then used directly (raw amplitude).
    new_sampling_rate = config["new_sampling_rate"]
    transform = torchaudio.transforms.Resample(
        orig_freq=config["original_sampling_rate"],
        new_freq=new_sampling_rate,
    ).to(device)

    # ----- Objective and optimizer ----------------------------------------
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # ----- Training log ----------------------------------------------------
    csvfile = open(filename, mode="a", newline="")
    fieldnames = ["Epoch", "Batch", "Loss"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # ----- Training loop ---------------------------------------------------
    for epoch in range(config["epochs"]):
        model.train()
        for i, (signals, labels) in enumerate(train_loader):
            signals, labels = signals.to(device), labels.to(device)

            signals = transform(signals)
            signals = utils.normalize(signals)

            batch_size = signals.shape[0]

            # raw amplitude: reshape to (seq_len, batch, n_inp=1)
            signals = signals.reshape(
                batch_size, 1, data_params["final_signal_length"]
            )
            input_signal = signals.permute(2, 0, 1)

            optimizer.zero_grad()
            output = model(input_signal)
            loss = objective(output, labels)
            loss.backward()
            optimizer.step()

            writer.writerow(
                {"Epoch": epoch, "Batch": i, "Loss": loss.item()}
            )

            print(
                "Experiment: {}, Epoch: {}, Batch: {}, Loss: {}".format(
                    experiment_id, epoch, i, loss.item()
                )
            )

        # ----- Save the model after every epoch ---------------------------
        path = "./results/{}/{}_epoch{}_experiment{}.pth".format(
            network, network, epoch, experiment_id
        )
        if config["save_model"]:
            torch.save(model.state_dict(), path)

        # ----- Evaluate on the test set -----------------------------------
        test_accuracy, test_loss = utils.evaluation(
            data_loader=test_loader,
            model=model,
            batch_size=config["batch"],
            objective=objective,
            device=device,
            transform=transform,
            data_params=data_params,
        )

        test_filename = "./results/{}/final_experiment{}.csv".format(
            network, experiment_id
        )
        network_name = "{}_epoch{}_experiment{}".format(
            network, epoch, experiment_id
        )

        utils.write_results(
            filename=test_filename,
            test_accuracy=test_accuracy,
            test_loss=test_loss,
            network=network_name,
        )

    csvfile.close()


if __name__ == "__main__":
    main()
