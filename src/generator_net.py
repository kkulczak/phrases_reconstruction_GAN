import torch
from torch import nn

from src.utils import softmax_gumbel_noise


class GeneratorNet(nn.Module):
    def __init__(self, config: dict, z_size: int = 20):
        super(GeneratorNet, self).__init__()
        self.n_feature = (
            config['concat_window'] * config['ascii_size']
            + config['gen_z_size']
        )
        self.n_out = config['ascii_size']
        self.config = config

        self.hidden1 = nn.Sequential(
            nn.Linear(
                self.n_feature,
                config['gen_hidden_size']
            ),
            nn.ReLU(),
            nn.Linear(config['gen_hidden_size'], self.n_out),
        )

    def forward(self, x, temperature: float = 0.9):
        batch_size = x.shape[0]
        x = x.reshape(
            batch_size * self.config['phrase_length'],
            self.config['concat_window'] * self.config['ascii_size']
        )

        if self.config['gen_z_size'] > 0:
            generator_seed = torch.rand(
                (batch_size, self.config['gen_z_size']),
                dtype=torch.float32,
                device=x.device,
            ).repeat_interleave(repeats=self.config['phrase_length'], dim=0)
            x = torch.cat([generator_seed, x], dim=1)

        x = self.hidden1(x)

        log_prob = x.reshape(
            batch_size,
            self.config['phrase_length'],
            self.config['ascii_size']
        )
        soft_prob = softmax_gumbel_noise(log_prob, temperature)

        return soft_prob
