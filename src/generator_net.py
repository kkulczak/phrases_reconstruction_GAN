import torch
from torch import nn

from src.utils import softmax_gumbel_noise


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
        x = x.reshape(
            -1,
            self.config['phrase_length'] * self.config['concat_window'] *
            self.config['ascii_size']
        )
        batch_size = x.shape[0]

        # generator_seed = torch.rand(
        #     (batch_size, self.config['gen_z_size']),
        #     dtype=torch.float32,
        #     device=x.device,
        # )
        # TODO remove replacing ranodm by zeros
        generator_seed = torch.zeros(
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
        soft_prob = softmax_gumbel_noise(log_prob, temperature)

        return soft_prob
