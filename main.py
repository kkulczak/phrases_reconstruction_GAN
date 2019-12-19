import torch
import yaml
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data_loader import (
    AmericanNationalCorpusDataset,
    ObliterateLetters,
    ToTensor,
)
from src.discriminator_net import DiscriminatorNet
from src.generator_net import GeneratorNet
from src.utils import (
    generate_inter_sample,
    compute_gradient_penalty,
)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    config = yaml.safe_load(open('config.yaml', 'r'))
    print(config)
    noisy_phrases = AmericanNationalCorpusDataset(
        config,
        transform_raw_phrase=ObliterateLetters(obliterate_ratio=0.1),
        transform_sample_dict=ToTensor()
    )
    real_phrases = AmericanNationalCorpusDataset(
        config,
        transform_raw_phrase=None,
        transform_sample_dict=ToTensor()
    )

    print('Dataset_size:', len(noisy_phrases))
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

    np.random.seed(0)

    generator = GeneratorNet(config).to(device)
    discriminator = DiscriminatorNet(config).to(device)
    for epoch in (range(5)):
        for noisy_batch, real_batch in tqdm(zip(noisy_data_loader, real_data_loader)):
            gen_input = noisy_batch['concat_phrase'].to(device)
            # for x in bath_data['raw_phrase']:
            #     print(train_dataset.show(x))
            fake_sample = generator.forward(gen_input)
            real_sample = real_batch['raw_phrase'].float().to(device)
            inter_sample = generate_inter_sample(fake_sample, real_sample)



            # for x in fake_sample:
            #     print(train_dataset.show(x))
            # print(fake_sample.shape)
            fake_sample_pred = discriminator.forward(fake_sample)
            real_sample_pred = discriminator.forward(real_sample)
            inter_sample_pred = discriminator.forward(inter_sample)

            gradient_penalty = compute_gradient_penalty(
                output=inter_sample_pred,
                input=inter_sample
            )

            # exit()

    # for sample in train_dataset:
    #     sample = sample['concat_phrase'].to(device)
    #     print(sample.shape)

    # fake_sample = generator.forward(noisy_batch['concat_phrase'])
    #
    # print(fake_sample.shape)

    # _id = np.random.randint(0, len(train_dataset))
    # x = train_dataset[_id]
    # print(train_dataset.show(x))
    # for i in x['concat_phrase']:
    #     print(AmericanNationalCorpusDataset.show(i.reshape(-1, 256)))
