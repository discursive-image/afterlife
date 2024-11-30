import torch
import torch.nn as nn
import json
from safetensors.torch import load_file
from pathlib import Path


class TrajectoryPredictor(nn.Module):
    def __init__(
        self,
        input_dim=2,
        hidden_dim=128,
        lstm_layers=3,
        dense_layers=[64, 32, 16],
        output_dim=2,
    ):
        super(TrajectoryPredictor, self).__init__()

        # LSTM layers for temporal processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # Fully connected layers for further feature processing
        dense_layers_list = []
        input_size = hidden_dim
        for layer_size in dense_layers:
            dense_layers_list.append(nn.Linear(input_size, layer_size))
            dense_layers_list.append(nn.ReLU())
            input_size = layer_size
        self.fc = nn.Sequential(*dense_layers_list)

        # Output layer with Sigmoid activation
        self.output_layer = nn.Linear(input_size, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, input_sequence_length, input_dim)
        lstm_out, _ = self.lstm(
            x
        )  # lstm_out: (batch_size, input_sequence_length, hidden_dim)
        dense_out = self.fc(
            lstm_out
        )  # Pass each timestep's hidden state through dense layers
        output = self.sigmoid(
            self.output_layer(dense_out)
        )  # Output: (batch_size, input_sequence_length, output_dim)
        return output


def load_from_path(basedir: str) -> TrajectoryPredictor:
    basedir = Path(basedir)
    config_path = basedir.joinpath("config.json")
    config_fd = open(config_path, "r")
    config = json.load(config_fd)

    model = TrajectoryPredictor(
        lstm_layers=config["lstm_layers"]["value"],
        dense_layers=config["dense_layers"]["value"],
        hidden_dim=config["hidden_dim"]["value"],
    )

    state_path = basedir.joinpath("model.safetensors")
    state_dict = load_file(state_path)
    model.load_state_dict(state_dict)

    return model
