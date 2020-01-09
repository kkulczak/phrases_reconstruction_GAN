import logging
import os
from shutil import copyfile

import numpy as np
import torch

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import (
    AmericanNationalCorpusDataset,
    ObliterateLetters,
    ToTensor,
)
from src.discriminator_net import DiscriminatorNet
from src.eval import measure_accuracy
from src.generator_net import GeneratorNet
from src.penalties import compute_gradient_penalty


def train(config, save_dir, device='cpu'):
    noisy_phrases = AmericanNationalCorpusDataset(
        config,
        transform_raw_phrase=ObliterateLetters(
            obliterate_ratio=config['replace_with_noise_probability']
        ),
        transform_sample_dict=ToTensor()
    )
    real_phrases = AmericanNationalCorpusDataset(
        config,
        transform_raw_phrase=None,
        transform_sample_dict=ToTensor()
    )

    logging.info(f'Dataset_size: {len(noisy_phrases)}')

    noisy_data_loader = DataLoader(
        noisy_phrases,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True
    )
    real_data_loader = DataLoader(
        real_phrases,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True
    )

    test_noisy_data_loader = DataLoader(
        noisy_phrases,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False
    )

    test_real_data_loader = DataLoader(
        real_phrases,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False
    )

    generator = GeneratorNet(config).to(device)
    discriminator = DiscriminatorNet(config).to(device)

    optimizer_gen = optim.Adam(
        generator.parameters(),
        lr=config['learning_rate'],
        betas=(0.5, 0.9),
    )
    optimizer_dis = optim.Adam(
        discriminator.parameters(),
        lr=config['learning_rate'],
        betas=(0.5, 0.9),
    )

    if save_dir is not None:
        writer = SummaryWriter(save_dir)
        copyfile('config.yaml', os.path.join(save_dir, 'config.yaml'))

    for epoch in range(config['num_epochs']):
        logging.info(f'Epoch: {epoch + 1}')
        logs = []
        for step, (noisy_batch, real_batch) in tqdm(
            enumerate(zip(noisy_data_loader, real_data_loader)),
            total=int(len(noisy_phrases) / config['batch_size'])
        ):

            #### Discriminator_training
            discriminator.zero_grad()

            gen_input = noisy_batch['concat_phrase'].to(device)

            real_sample = real_batch['raw_phrase'].float().to(device)
            fake_sample = generator.forward(gen_input, temperature=0.9)

            fake_sample_pred = discriminator.forward(
                fake_sample.detach()
            )
            real_sample_pred = discriminator.forward(real_sample)

            gradient_penalty = compute_gradient_penalty(
                discriminator,
                real_sample,
                fake_sample
            )

            fake_score = fake_sample_pred.mean()
            real_score = real_sample_pred.mean()

            dis_loss = (
                - (real_score - fake_score)
                # + config['gradient_penalty_ratio'] * gradient_penalty
            )

            dis_loss.backward()

            optimizer_dis.step()

            #### Generator traning
            generator.zero_grad()
            # Not needed???
            # fake_sample = generator.forward(gen_input, temperature=0.9)
            fake_sample_pred = discriminator.forward(
                fake_sample
            )
            fake_score = fake_sample_pred.mean()

            gen_loss = - (fake_score - real_score.item())

            gen_loss.backward()
            optimizer_gen.step()

            # Logging values
            logs.append([
                dis_loss.item(),
                gen_loss.item(),
                gradient_penalty.item(),
                real_score.item(),
                fake_score.item(),
            ])

        # Traning supervision. Saving scalars for tensorboard
        epoch_logs = np.array(logs)
        logs_labels = [
            'loss/dis_loss',
            'loss/gen_loss',
            'gradient_penalty',
            'scores/real_score',
            'scores/fake_score',
        ]
        if save_dir is not None:
            for label, x in zip(logs_labels, epoch_logs.T):
                writer.add_scalars(
                    label,
                    {
                        'min':  x.min(),
                        'mean': x.mean(),
                        'max':  x.max(),
                    },
                    global_step=epoch + 1
                )
                writer.add_histogram(
                    f'hist_{label}',
                    x
                )
        if (
            epoch % config['eval_epoch_every'] == 0
            or epoch + 1 == config['num_epochs']
        ):
            acc = measure_accuracy(
                generator,
                test_real_data_loader,
                test_noisy_data_loader,
                device
            )
            logging.info(f'Accuracy: {acc:.2f}')
            if save_dir is not None:
                writer.add_scalar(
                    'accuracy/accuracy',
                    acc,
                    global_step=epoch + 1
                )

        # Model saving
        if save_dir is not None:
            torch.save(
                generator.state_dict(),
                os.path.join(save_dir, f'epoch_{epoch + 1}_generator.pt')
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(save_dir, f'epoch_{epoch + 1}_discriminator.pt')
            )
    if save_dir is not None:
        writer.close()
