import torchtext
import spacy
import os
import csv
import json
import random
from torchtext.data import Field, RawField


class Dataset:
    def __init__(self):
        tokenize = lambda x: x.split()
        INPUT = Field(sequential=True, batch_first=True, tokenize=tokenize, lower=True)
        TGT = Field(sequential=True, batch_first=True)
        SHOW_INP = RawField()
        fields = [('tgt', TGT), ('input', INPUT), ('show_inp', SHOW_INP), ]


if __name__ == '__main__':
    load_data()

