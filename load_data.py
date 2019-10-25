import torchtext
import spacy
import os
import json
import random
from torchtext.data import Field


class NLP:
    def __init__(self):
        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def word_tokenize(self, text, lower=False):  # create a tokenizer function
        if text is None: return text
        text = ' '.join(text.split())
        if lower: text = text.lower()
        toks = [tok.text for tok in self.nlp.tokenizer(text)]
        return ' '.join(toks)


class Dataset:
    def __init__(self):
        tokenize = lambda x: x.split()
        INPUT = Field(sequential=True, batch_first=True, tokenize=tokenize, lower=True)
        TARGET = Field(sequential=True, batch_first=True)


def load_data(data_dir='./data'):
    train_valid_path = os.path.join(data_dir, 'TRAIN_FILE.TXT')
    test_path = os.path.join(data_dir, 'TEST_FILE_FULL.TXT')

    train_valid_data = preprocess(train_valid_path)
    test_data = preprocess(test_path)

    data = dict()
    data['train'], data['valid'] = train_valid_split(train_valid_data, train_rate=0.8)
    data['test'] = test_data

    for key, value in data.items():
        json_fname = os.path.join(data_dir, '{}.json'.format(key))
        json_to_write = json.dumps(value, indent=4)
        with open(json_fname, 'w') as f:
            f.write(json_to_write)


def train_valid_split(data, train_rate=0.8):
    random.shuffle(data)
    train = data[:int(len(data)*train_rate)]
    valid = data[int(len(data)*train_rate):]
    return train, valid


def preprocess(path):
    ENT_1_START = '<e1>'
    ENT_1_END = '</e1>'
    ENT_2_START = '<e2>'
    ENT_2_END = '</e2>'

    nlp = NLP()
    data = []
    with open(path) as f:
        lines = [line.strip() for line in f]
    for idx in range(0, len(lines), 4):
        id = int(lines[idx].split("\t")[0])
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        sentence = sentence.strip()

        sentence = sentence.replace(ENT_1_START, ' ENT_1_START ')
        sentence = sentence.replace(ENT_1_END, ' ENT_1_END ')
        sentence = sentence.replace(ENT_2_START, ' ENT_2_START ')
        sentence = sentence.replace(ENT_2_END, ' ENT_2_END ')

        sentence = nlp.word_tokenize(sentence)

        ent1 = sentence.split(' ENT_1_START ')[-1].split(' ENT_1_END ')[0]
        ent2 = sentence.split(' ENT_2_START ')[-1].split(' ENT_2_END ')[0]

        data.append({
            'label': relation,
            'sentence': sentence,
            'ent1': ent1,
            'ent2': ent2,
            'id': id,
        })

    return data


if __name__ == '__main__':
    load_data()

