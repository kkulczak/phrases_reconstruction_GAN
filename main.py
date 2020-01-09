import datetime
import logging
import os
from argparse import ArgumentParser
from shutil import copyfile

import torch
import yaml
import numpy as np
from torch import optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data_loader import (
    AmericanNationalCorpusDataset,
    ObliterateLetters,
    ToTensor,
)
from src.discriminator_net import DiscriminatorNet
from src.eval import (
    show_examples,
    eval_with_mean_accuracy,
)
from src.generator_net import GeneratorNet
from src.train import train
from src.utils import (
    generate_inter_sample,
)
from src.penalties import compute_gradient_penalty

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'mode',
        choices=['train', 'eval', 'show']
    )
    parser.add_argument(
        '--epoch-num', '-e',
        type=int,
        help='Epoch number to be restored'
    )
    parser.add_argument(
        '--path-to-checkpoints', '-p',
        type=str,
        help='Path to directory with saved model checkpoints'
    )
    parser.add_argument(
        '--device', '-d',
        default='gpu',
        choices=['gpu', 'cpu'],
    )
    parser.add_argument(
        '--disable-logs',
        action='store_true',
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG)

    if not args.disable_logs:
        mydir = os.path.join(
            os.getcwd(),
            'data',
            datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        )
    else:
        mydir = None

    if args.device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device
    logging.info(f'device: {device}')

    if args.path_to_checkpoints is not None:
        config_path = os.path.join(args.path_to_checkpoints, 'config.yaml')
    else:
        config_path = 'config.yaml'
    config = yaml.safe_load(open(config_path, 'r'))
    logging.info(f'config file loaded from: {config_path}')

    if args.mode == 'train':
        train(
            config,
            mydir,
            device=device
        )
        exit(0)
    if not (
        isinstance(args.epoch_num, int) and
        isinstance(args.path_to_checkpoints, str)
    ):
        raise ValueError('Required arguments not given')

    if args.mode == 'eval':
        eval_with_mean_accuracy(args, config, device)
    if args.mode == 'show':
        show_examples(args, config, device=device)
