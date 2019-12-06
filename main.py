import torch
import yaml
import numpy as np
from torch.utils.data.dataloader import DataLoader

from data_loader import (
    AmericanNationalCorpusDataset,
    ObliterateLetters,
    ToTensor,
)
from src.generator_net import GeneratorNet

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    config = yaml.safe_load(open('config.yaml', 'r'))
    print(config)
    train_dataset = AmericanNationalCorpusDataset(
        dataset_size=config['dataset_size'],
        phrase_length=config['phrase_length'],
        concat_window=config['concat_window'],
        ascii_size=config['ascii_size'],
        transform_raw_phrase=ObliterateLetters(obliterate_ratio=0.15),
        transform_sample_dict=ToTensor()
    )
    print('Dataset_size:', len(train_dataset))
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True
    )

    np.random.seed(0)

    generator = GeneratorNet(config).to(device)
    for bath_data in train_data_loader:
        batch = bath_data['concat_phrase']
        fake_sample = generator.forward(batch).detach()
        for x in fake_sample:
            print(train_dataset.show(x))


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
