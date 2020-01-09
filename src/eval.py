import logging
import operator

import itertools
import os
import numpy as np
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import reduce

from data_loader import (
    AmericanNationalCorpusDataset,
    ObliterateLetters,
    ToTensor,
)
from src.discriminator_net import DiscriminatorNet
from src.generator_net import GeneratorNet


def load_models(args, config, device='cpu') -> Tuple[
    GeneratorNet, DiscriminatorNet]:
    generator = GeneratorNet(config).to(device)
    generator.load_state_dict(
        torch.load(
            os.path.join(
                args.path_to_checkpoints,
                f'epoch_{args.epoch_num}_generator.pt'
            ),
            map_location=device
        )
    )
    generator.eval()

    discriminator = DiscriminatorNet(config).to(device)
    discriminator.load_state_dict(
        torch.load(
            os.path.join(
                args.path_to_checkpoints,
                f'epoch_{args.epoch_num}_discriminator.pt'
            ),
            map_location=device
        )
    )
    discriminator.eval()

    return generator, discriminator


def show_examples(args, config, device='cpu'):
    generator, _ = load_models(args, config, device=device)

    noisy_phrases = AmericanNationalCorpusDataset(
        config,
        transform_raw_phrase=ObliterateLetters(
            obliterate_ratio=config['replace_with_noise_probability']
        ),
        transform_sample_dict=ToTensor()
    )

    noisy_data_loader = DataLoader(
        noisy_phrases,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )

    with torch.no_grad():
        for x in itertools.islice(noisy_data_loader, 5):
            _input = x['concat_phrase'].to(device)
            out = generator.forward(_input).cpu()
            print('#' * 40)
            print(noisy_phrases.show(x['raw_phrase']))
            print(noisy_phrases.show(out))
            print('#' * 40)


def measure_accuracy(generator, real_data_loader, fake_data_loader, device):
    correct = 0
    elements = 0
    with torch.no_grad():
        for fake_batch, real_batch in tqdm(
            zip(fake_data_loader, real_data_loader)):
            _input = fake_batch['concat_phrase'].to(device)
            output = generator.forward(_input)

            correct += np.sum(
                np.argmax(output.detach().cpu().numpy(), axis=-1)
                == np.argmax(real_batch['raw_phrase'].numpy(), axis=-1)
            )
            elements += reduce(
                operator.mul,
                real_batch['raw_phrase'].shape[:-1],
                1
            )
    logging.debug(f'{correct} {elements} {correct / elements}')
    return correct / elements


def eval_with_mean_accuracy(args, config, device):
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
    generator, _ = load_models(args, config, device)
    acc = measure_accuracy(
        generator,
        test_real_data_loader,
        test_noisy_data_loader,
        device
    )
    print(f'Mean Accuracy: {acc:.2f}')
