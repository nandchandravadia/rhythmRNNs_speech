from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class coRNNCell(nn.Module):
    def __init__(self, network_type, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNNCell, self).__init__()

        # network parameters
        self.network_type = network_type
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon

        # define activation fxn
        activation = "tanh"
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sin":
            self.activation = torch.sin
        else:
            raise NotImplementedError

        # input weights
        self.I_ext = nn.Linear(in_features=n_inp, out_features=n_hid, bias=True)

        # recurrent weights of hidden states
        self.R = nn.Linear(in_features=n_hid, out_features=n_hid, bias=False)

        # recurrent weights of velocity of hidden states
        self.F = nn.Linear(in_features=n_hid, out_features=n_hid, bias=False)

    def forward(self, x, hy, hz):

        # our update equations
        alpha = 1
        activation = self.activation(
            alpha * (self.R(hy) + self.F(hz) + self.I_ext(x))
        )

        hz = hz + self.dt * (activation - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz, activation


class coRNN(nn.Module):
    def __init__(self, network_type, n_inp, n_hid, n_out, dt, gamma, epsilon):
        super(coRNN, self).__init__()

        # network parameters
        self.n_hid = n_hid
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_out = n_out

        self.cell = coRNNCell(network_type, n_inp, n_hid, dt, self.gamma, self.epsilon)
        self.readout = nn.Linear(n_hid, n_out)

    def forward(self, x):

        # initialize hidden states
        hy = torch.zeros(x.size(1), self.n_hid, device=device)
        hz = torch.zeros(x.size(1), self.n_hid, device=device)

        # roll the recurrence forward over the full sequence
        for t in range(x.size(0)):
            hy, hz, activation = self.cell(x[t], hy, hz)

        # read out the final hidden state only
        output = self.readout(hy)

        return output
