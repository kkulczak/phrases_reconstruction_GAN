import torch
from torch.autograd import Variable
from torch import nn
import numpy as np


class GeneratorNet(nn.Module):
    def __init__(self, config: dict, z_size: int = 20):
        super(GeneratorNet, self).__init__()
        self.n_feature = config['phrase_length'] \
                         * config['concat_window'] \
                         * config['ascii_size'] \
                         + config['gen_z_size']
        self.n_out = config['phrase_length'] * config['ascii_size']
        self.config = config

        self.hidden1 = nn.Sequential(
            nn.Linear(
                self.n_feature,
                512
            ),
            nn.ReLU()
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, self.n_out),
        )

    def forward(self, x, temperature: float = 0.9):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        generator_seed = torch.rand(
            (batch_size, self.config['gen_z_size']),
            dtype=torch.float32,
            device=x.device,
        )

        x = torch.cat([generator_seed, x], dim=1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        log_prob = x.reshape(
            batch_size,
            self.config['phrase_length'],
            self.config['ascii_size']
        )
        soft_prob = self.softmax_gumbel_noise(log_prob, temperature)

        return soft_prob

    @staticmethod
    def softmax_gumbel_noise(
        logits: torch.Tensor,
        temperature: float,
        eps: float = 1e-20
    ):
        U = torch.rand(logits.shape, device=logits.device)
        noise = -torch.log(-torch.log(U + eps) + eps)
        y = logits + noise
        return nn.functional.softmax(y / temperature, dim=-1)
