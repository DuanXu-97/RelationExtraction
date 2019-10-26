import argparse
import torch
from load_data import *
from model import *
from hyperparameters import *
from model import AttBiLSTM
from eval import *
import load_data as ld


def train(args):

    if args.use_gpu == 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.model == 'AttBiLSTM':
        config = AttBiLSTMHP()
        model = AttBiLSTM(config)
    else:
        config = None
        model = None

    dataset = Dataset(data_dir=args.save_dir, train_fname=args.train_fname,
                      valid_fname=args.valid_fname, test_fname=args.test_fname,)

    train_dl, valid_dl, test_dl = dataset.get_dataloader(batch_size=config.batch_size, device=device)

    validator = Validator(dataloader=valid_dl, save_dir=args.save_dir,
                          save_log_fname=args.save_log_fname,
                          save_model_fname=args.save_model_fname,
                          valid_or_test='valid',
                          vocab_itos=dataset.INPUT.vocab.itos,
                          label_itos=dataset.TGT.vocab.itos)
    tester = Validator(dataloader=test_dl, save_log_fname=args.save_log_fname,
                       save_dir=args.save_dir, valid_or_test='test',
                       vocab_itos=dataset.INPUT.vocab.itos,
                       label_itos=dataset.TGT.vocab.itos)
    predictor = Predictor(args.save_vocab_fname)

    model = model.to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='AttBiLSTM', help="The model to be trained")
    parser.add_argument('--use_gpu', type=int, default=1, help="whether use gpu")
    parser.add_argument('--train_fname', type=str, default='train.csv', help="the filename of training data")
    parser.add_argument('--valid_fname', type=str, default='valid.csv', help="the filename of validation data")
    parser.add_argument('--test_fname', type=str, default='test.csv', help="the filename of test data")



    args = parser.parse_args()
    train(args)