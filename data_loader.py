import logging
import os
import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from nltk.corpus import masc_tagged
import nltk

GAP_CHARACTER = '\0'


class AmericanNationalCorpusDataset(Dataset):
    """
        The Open American National Corpus Dataset
    """
    source_data = None

    def __init__(
        self,
        config,
        transform_raw_phrase: callable = None,
        transform_sample_dict: callable = None
    ):
        # Download corpus
        if os.path.join(os.getcwd(), 'raw_phrases') not in nltk.data.path:
            nltk.data.path.append(os.path.join(os.getcwd(), 'raw_phrases'))
        try:
            nltk.data.find('corpora/masc_tagged.zip')
        except LookupError as e:
            logging.info(
                'Corpus not found. Downloading american national corpus'
            )
            nltk.download(
                'masc_tagged',
                download_dir='raw_phrases',
                quiet=True,
            )
        self.phrase_length = config['phrase_length']
        self.concat_window = config['concat_window']
        self.ascii_size = config['ascii_size']
        self.transform_raw_phrase = transform_raw_phrase
        self.transform_sample_dict = transform_sample_dict
        self.concat_window_indexes = (
            (np.arange(self.concat_window) - self.concat_window // 2)[None, :]
            + (np.arange(self.phrase_length))[:, None]
        )
        self.concat_window_indexes[self.concat_window_indexes < 0] = 0
        self.concat_window_indexes[
            self.concat_window_indexes >= self.phrase_length
        ] = self.phrase_length - 1
        self.raw_phrases = [
            processed
            for phrase in masc_tagged.sents()[:config['dataset_size']]
            if len(phrase) > 4 and len(phrase[0]) < 30
            for processed in [self.preprocess_phrase(phrase)]
            if processed is not None
        ]

        self.raw_phrases = np.array(
            self.raw_phrases
        )

        self.raw_phrases = one_hot(
            torch.from_numpy(self.raw_phrases),
            num_classes=self.ascii_size
        ).numpy()

    def build_windowed_phrase(self, feat):
        return feat[self.concat_window_indexes].reshape(
            self.phrase_length,
            self.concat_window * self.ascii_size
        ).astype(np.float32)

    def preprocess_phrase(self, phrase):
        lengths = np.array([len(x) for x in phrase])
        lengths[1:] += 1
        cum_sum = np.cumsum(lengths)
        last_word_id = np.searchsorted(
            cum_sum > self.phrase_length,
            1,
            side='left'
        )
        processed = ' '.join(phrase[:last_word_id]).lower()

        _ascii = np.array([int(ord(x)) for x in processed], dtype=int)

        if (_ascii > self.ascii_size).any():
            return None
        if (_ascii == ord(GAP_CHARACTER)).any():
            return None

        _ascii = np.concatenate(
            (_ascii,
            ord(GAP_CHARACTER) * np.ones(
                self.phrase_length - _ascii.size,
                dtype=int)
            )
        )

        if (_ascii == ord(GAP_CHARACTER)).all():
            breakpoint()

        return _ascii

    def __len__(self):
        return self.raw_phrases.shape[0]

    def __getitem__(self, item):
        sample = self.raw_phrases[item]
        if self.transform_raw_phrase is not None:
            sample = self.transform_raw_phrase(sample)

        sample_dict = {
            'raw_phrase':    sample,
            'concat_phrase': self.build_windowed_phrase(
                sample
            ),
        }
        if self.transform_sample_dict is not None:
            sample_dict = self.transform_sample_dict(sample_dict)

        return sample_dict

    def show(self, sample):
        if isinstance(sample, dict):
            xs = sample['raw_phrase'].numpy()
        elif isinstance(sample, np.ndarray):
            xs = sample
        elif isinstance(sample, torch.Tensor):
            xs = sample.numpy()
        else:
            raise ValueError('sample must be dict or array or torch.Tensor')
        xs = xs.reshape(-1, self.ascii_size)
        return ''.join(
            chr(x) if x != ord(GAP_CHARACTER) else "_"
                for x in np.argmax(xs, axis=1)
        )


class ObliterateLetters(object):
    def __init__(self, obliterate_ratio: float):
        self.obliterate_ratio = obliterate_ratio

    def __call__(self, sample):
        ids = np.random.uniform(size=sample.shape[:-1]) < self.obliterate_ratio
        one_hot_gap_character = np.zeros(sample.shape[-1], dtype=sample.dtype)
        one_hot_gap_character[ord(GAP_CHARACTER)] = 1
        sample[ids, :] = one_hot_gap_character
        return sample


class ToTensor(object):
    def __call__(self, sample_dict):
        return {k: torch.from_numpy(v) for k, v in sample_dict.items()}
