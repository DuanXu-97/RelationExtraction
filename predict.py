import json
import torch
import argparse

from model import AttBiLSTM
from load_data import Dataset
from preprocess import sentence_preprocess


class Predictor:
    def __init__(self, vocab_fname):
        with open(vocab_fname) as f:
            vocab = json.load(f)
        self.tgt_itos = vocab['tgt_vocab']['itos']
        self.input_stoi = vocab['input_vocab']['stoi']
        self.ent_stoi = vocab['ent_vocab']['stoi']
        self.dataset = Dataset()

    def use_pretrained_model(self, model_fname, device=torch.device('cpu')):
        self.device = device
        checkpoint = torch.load(model_fname)
        model_config = checkpoint['config']
        model = AttBiLSTM(model_config)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        model.eval()
        self.model = model

    def pred_sent(self, test_sentence, model=None):
        INPUT_field = self.dataset.INPUT
        ENT_field = self.dataset.ENT

        test_sentence, test_ent1, test_ent2 = sentence_preprocess(test_sentence)

        print("sentence: ", test_sentence)
        print("ent1: ", test_ent1)
        print("ent2: ", test_ent2)

        if model is None: model = self.model
        model.eval()
        device = next(model.parameters()).device
        input_stoi = self.input_stoi
        ent_stoi = self.ent_stoi

        test_sen_ixs = INPUT_field.preprocess(test_sentence)
        test_sen_ixs = [[input_stoi[x] if x in input_stoi else 0
                         for x in test_sen_ixs]]
        test_ent1_ixs = ENT_field.preprocess(test_ent1)
        test_ent2_ixs = ENT_field.preprocess(test_ent2)
        test_ent1_ixs = [[ent_stoi[x] if x in ent_stoi else 0
                         for x in test_ent1_ixs]]
        test_ent2_ixs = [[ent_stoi[x] if x in ent_stoi else 0
                         for x in test_ent2_ixs]]

        with torch.no_grad():
            test_sen_batch = torch.LongTensor(test_sen_ixs).to(device)
            test_ent1_batch = torch.LongTensor(test_ent1_ixs).to(device)
            test_ent2_batch = torch.LongTensor(test_ent2_ixs).to(device)

            print(test_ent1_batch.shape)
            print(test_ent2_batch.shape)

            output = model.predict(test_sen_batch, test_ent1_batch, test_ent2_batch)
            prediction = self.tgt_itos[output[0]]

            return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./tmp/model', help="The model to be used")
    parser.add_argument('--use_gpu', type=int, default=1, help="whether use gpu")
    parser.add_argument('--vocab_fname', type=str, default='vocab.json', help="vocab filename")
    parser.add_argument('--input', type=str, default='The most common <e1>audits</e1> were about <e2>waste</e2> and recycling.', help="input sentence")

    args = parser.parse_args()

    if args.use_gpu == 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    predictor = Predictor(args.vocab_fname)
    predictor.use_pretrained_model(args.model_path, device=device)
    prediction = predictor.pred_sent(args.input)

    print(prediction)


