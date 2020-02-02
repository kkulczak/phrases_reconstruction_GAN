import collections
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


class BatchSampler:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(data_loader)

    def sample_batch(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            return next(self.iter)


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
        shuffle=True,
        drop_last=True,
    )
    real_data_loader = DataLoader(
        real_phrases,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        drop_last=True,
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
        lr=config['gen_learning_rate'],
        betas=(0.5, 0.9),
    )
    optimizer_dis = optim.Adam(
        discriminator.parameters(),
        lr=config['dis_learning_rate'],
        betas=(0.5, 0.9),
    )

    if save_dir is not None:
        writer = SummaryWriter(save_dir)
        copyfile('config.yaml', os.path.join(save_dir, 'config.yaml'))

    noisy_sampler = BatchSampler(noisy_data_loader)
    real_sampler = BatchSampler(real_data_loader)

    logs = collections.defaultdict(list)
    for step in range(1, config['steps'] + 1):
        for _ in range(config['dis_iter']):
            #### Discriminator_training
            discriminator.zero_grad()
            noisy_batch = noisy_sampler.sample_batch()
            real_batch = real_sampler.sample_batch()

            gen_input = noisy_batch['concat_phrase'].to(device)
            real_sample = real_batch['raw_phrase'].float().to(device)
            fake_sample = generator.forward(gen_input, temperature=0.9)

            fake_sample_pred = discriminator.forward(
                fake_sample
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
                fake_score - real_score
                + config['gradient_penalty_ratio'] * gradient_penalty
            )

            dis_loss.backward()
            optimizer_dis.step()

            logs['gradient_penalty'].append(gradient_penalty.item())
            logs['real_score'].append(real_score.item())
            logs['fake_score'].append(fake_score.item())
            logs['dis_loss'].append(dis_loss.item())

        for _ in range(config['gen_iter']):
            #### Generator traning
            generator.zero_grad()
            noisy_batch = noisy_sampler.sample_batch()
            # real_batch = real_sampler.sample_batch()
            gen_input = noisy_batch['concat_phrase'].to(device)
            # real_sample = real_batch['raw_phrase'].float().to(device)
            # real_sample_pred = discriminator.forward(real_sample)
            # real_score = real_sample_pred.mean()

            fake_sample = generator.forward(gen_input, temperature=0.9)
            fake_sample_pred = discriminator.forward(
                fake_sample
            )
            fake_score = fake_sample_pred.mean()
            gen_loss = - fake_score
            gen_loss.backward()
            optimizer_gen.step()

            logs['real_score'].append(real_score.item())
            logs['fake_score'].append(fake_score.item())
            logs['gen_loss'].append(gen_loss.item())

        # Traning supervision. Saving scalars for tensorboard
        if step % config['print_step'] == 0:
            ###################################
            # Injected Prinitng
            ###################################
            print('#' * 80)
            p_fake = fake_sample.argmax(axis=-1)
            p_real = noisy_batch['raw_phrase'].argmax(axis=-1)

            def decode(xs):
                return ''.join(
                    "_"
                    if x == ord('\0') or x <= 31 or x > 126
                    else chr(x)
                        for x in xs
                )

            for (rl, fk) in zip(p_real[:5], p_fake[:5]):
                print(decode(rl))
                print(decode(fk))
            print('#' * 80)
            step_fer = (np.array(p_fake.cpu()) == np.array(p_real)).mean()
            print(f'Acc: {step_fer:.2}')
            ###################################
            # Injected Prinitng
            ###################################
            for k in logs.keys():
                logs[k] = np.array(logs[k]).mean()
            logging.info(
                f'Step:{step:5d} gen_loss:{logs["gen_loss"]:.3f} '
                f'dis_loss:{logs["dis_loss"]:.3f}'
            )

            def assign_label(x):
                if 'score' in x:
                    return f'scores/{x}'
                if 'loss' in x:
                    return f'loss/{x}'
                return x

            if save_dir is not None:
                for k, v in logs.items():
                    writer.add_scalar(
                        assign_label(k),
                        v,
                        global_step=step,
                    )
            logs.clear()
        if (
            step % config['eval_step'] == 0
            or step == config['steps']
        ):
            acc = measure_accuracy(
                generator,
                test_real_data_loader,
                test_noisy_data_loader,
                device
            )
            logging.info(f'EvalStep:{step:5d} Accuracy: {acc:.2f}')
            if save_dir is not None:
                writer.add_scalar(
                    'accuracy/accuracy',
                    acc,
                    global_step=step
                )
                # Model saving
                torch.save(
                    generator.state_dict(),
                    os.path.join(save_dir, f'epoch_{step}_generator.pt')
                )
                torch.save(
                    discriminator.state_dict(),
                    os.path.join(save_dir,
                                 f'epoch_{step}_discriminator.pt')
                )
    if save_dir is not None:
        writer.close()
