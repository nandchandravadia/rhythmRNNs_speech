"""
Configuration loading
"""

import json
import torch
from torch.distributions.uniform import Uniform


def load_config(path, experiment_id):
    """
    Load a single experiment config from a JSON file that contains one or
    more experiments keyed by experiment_id at the top level.
    """
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
    """
    Resolve a per-node parameter (e.g. "gamma" or "epsilon") into a tensor
    of shape [1, n_hid], based on flat config fields:

        <name>_mode : "constant" or "uniform"
        <name>_value           (constant mode)
        <name>_low, <name>_high (uniform mode)
    """
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
