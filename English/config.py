"""
Configuration loading
"""

import json
import torch
from torch.distributions.uniform import Uniform


def load_config(path, experiment_id):
    with open(path, "r") as f:
        data = json.load(f)

    key = str(experiment_id)

    if key not in data:
        raise KeyError(
            "Experiment {} not found in {}. Available: {}".format(
                experiment_id, path, sorted(data.keys())
            )
        )

    return data[key]


def resolve_parameter(config, name, n_hid, device):
    mode = config["{}_mode".format(name)]

    if mode == "constant":
        param = config["{}_value".format(name)] * torch.ones(size=[1, n_hid])

    elif mode == "uniform":
        low = config["{}_low".format(name)]
        high = config["{}_high".format(name)]
        param = Uniform(low=low, high=high).rsample(sample_shape=[1, n_hid])

    else:
        raise ValueError("Unknown sampling mode for {}: {}".format(name, mode))

    return param.to(device)
