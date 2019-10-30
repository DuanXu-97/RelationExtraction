import spacy
import os
import csv
import json
import random


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


def gen_csv(json_data, csv_path):
    csv_data = list()
    csv_line = dict()
    for line in json_data:
        sentence = line['sentence']
        csv_line = {
            'tgt': line['label'],
            'input': sentence,
            'show_inp': sentence,
            'ent1': line['ent1'],
            'ent2': line['ent2'],
            'id': line['id'],
        }
        csv_data += [csv_line]
    with open(csv_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=csv_line.keys())
        writer.writeheader()
        writer.writerows(csv_data)


if __name__ == '__main__':
    data_dir = './data'
    train_valid_path = os.path.join(data_dir, 'TRAIN_FILE.TXT')
    test_path = os.path.join(data_dir, 'TEST_FILE_FULL.TXT')

    train_valid_data = preprocess(train_valid_path)
    test_data = preprocess(test_path)

    data = dict()
    data['train'], data['valid'] = train_valid_split(train_valid_data, train_rate=0.8)
    data['test'] = test_data

    train_csv_path = os.path.join(data_dir, 'train.csv')
    valid_csv_path = os.path.join(data_dir, 'valid.csv')
    test_csv_path = os.path.join(data_dir, 'test.csv')

    gen_csv(data['train'], train_csv_path)
    gen_csv(data['valid'], valid_csv_path)
    gen_csv(data['test'], test_csv_path)


