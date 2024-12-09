from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


def repeat_new_axis(tensor, rep, dim):
    reps = [1] * len(tensor.shape)
    reps.insert(dim, rep)
    return tensor.unsqueeze(dim).repeat(*reps)


def load_control_points(path=None):
    if path is None:
        path = f'{Path(__file__).parent.parent}/assets/panda/panda.npy'
    control_points = torch.from_numpy(np.load(path))
    control_points = torch.cat(
        [
            control_points[[-2, 0]].float(),
            torch.tensor([[0, 0, 0, 1]]).float(),
            control_points[[1, -1]].float(),
        ]
    ).T  # 4x5
    return control_points


def get_activation_fn(activation):
    return getattr(nn, activation)()


def to_gpu(dic):
    for key in dic:
        if isinstance(dic[key], torch.Tensor):
            dic[key] = dic[key].cuda()
        elif isinstance(dic[key], list):
            if isinstance(dic[key][0], torch.Tensor):
                for i in range(len(dic[key])):
                    dic[key][i] = dic[key][i].cuda()
            elif isinstance(dic[key][0], list):
                for i in range(len(dic[key])):
                    for j in range(len(dic[key][i])):
                        if isinstance(dic[key][i][j], torch.Tensor):
                            dic[key][i][j] = dic[key][i][j].detach().cuda()


def to_cpu(dic):
    for key in dic:
        if isinstance(dic[key], torch.Tensor):
            dic[key] = dic[key].detach().cpu()
        elif isinstance(dic[key], list):
            if isinstance(dic[key][0], torch.Tensor):
                for i in range(len(dic[key])):
                    dic[key][i] = dic[key][i].detach().cpu()
            elif isinstance(dic[key][0], list):
                for i in range(len(dic[key])):
                    for j in range(len(dic[key][i])):
                        if isinstance(dic[key][i][j], torch.Tensor):
                            dic[key][i][j] = dic[key][i][j].detach().cpu()


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers,
        activation="ReLU", dropout=0.
    ):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        layers = []
        for m, n in zip([input_dim] + h[:-1], h):
            layers.extend([nn.Linear(m, n), get_activation_fn(activation)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
