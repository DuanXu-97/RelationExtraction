import torchtext
import spacy
import os
import csv
import json
import random
from utils import fwrite
import torch
from torchtext.data import Field, RawField, TabularDataset, BucketIterator, Iterator
from torchtext.vocab import GloVe


def get_stop_words():
    file_object = open('./data/stop_words.txt')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words


class Dataset:
    def __init__(self, data_dir='./data', train_fname='train.csv', valid_fname='valid.csv', test_fname='test.csv',
                 vocab_fname='vocab.json'):

        stop_words = get_stop_words()

        tokenize = lambda x: x.split()
        INPUT = Field(sequential=True, batch_first=True, tokenize=tokenize, lower=True)
        ENT = Field(sequential=False, batch_first=True, lower=True)
        TGT = Field(sequential=True, batch_first=True)
        SHOW_INP = RawField()
        fields = [('tgt', TGT), ('input', INPUT), ('show_inp', SHOW_INP), ('ent1', ENT), ('ent2', ENT)]

        datasets = TabularDataset.splits(
            fields=fields,
            path=data_dir,
            format=train_fname.rsplit('.')[-1],
            train=train_fname,
            validation=valid_fname,
            test=test_fname,
            skip_header=True,
        )

        INPUT.build_vocab(*datasets, max_size=100000,
                          vectors=GloVe(name='6B', dim=100),
                          unk_init=torch.Tensor.normal_, )
        TGT.build_vocab(*datasets)
        ENT.build_vocab(*datasets)

        self.INPUT = INPUT
        self.ENT = ENT
        self.TGT = TGT
        self.train_ds, self.valid_ds, self.test_ds = datasets

        if vocab_fname:
            writeout = {
                'tgt_vocab': {
                    'itos': TGT.vocab.itos, 'stoi': TGT.vocab.stoi,
                },
                'input_vocab': {
                    'itos': INPUT.vocab.itos, 'stoi': INPUT.vocab.stoi,
                },
                'ent_vocab': {
                    'itos': ENT.vocab.itos, 'stoi': ENT.vocab.stoi,
                },
            }
            fwrite(json.dumps(writeout, indent=4), vocab_fname)

    def get_dataloader(self, batch_size, device=torch.device('cpu')):

        train_iter, valid_iter = BucketIterator.splits(
            (self.train_ds, self.valid_ds),
            batch_sizes=(batch_size, batch_size),
            sort_within_batch=True,
            sort_key=lambda x: len(x.input),
            device=device,
            repeat=False,
        )

        test_iter = Iterator(
            self.test_ds,
            batch_size=1,
            sort=False,
            sort_within_batch=False,
            device=device,
            repeat=False,
        )
        train_dl = BatchWrapper(train_iter)
        valid_dl = BatchWrapper(valid_iter)
        test_dl = BatchWrapper(test_iter)
        return train_dl, valid_dl, test_dl


class BatchWrapper:
    def __init__(self, iterator):
        self.iterator = iterator

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            yield batch

