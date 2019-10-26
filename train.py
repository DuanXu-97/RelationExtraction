import argparse
import torch
import pdb
from load_data import *
from model import *
from hyperparameters import *
from model import AttBiLSTM
from evaluate import Validator, Predictor
from torch import optim
from tqdm import tqdm


def train(args):

    if args.use_gpu == 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.model == 'AttBiLSTM':
        config = AttBiLSTMHP()
    else:
        config = None

    dataset = Dataset(data_dir=args.data_dir, train_fname=args.train_fname,
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

    if args.model == 'AttBiLSTM':
        config.num_classes = len(dataset.TGT.vocab)
        model = AttBiLSTM(config)
    else:
        model = None

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    if args.load_model:
        predictor.use_pretrained_model(args.load_model, device=device)
        pdb.set_trace()
        predictor.pred_sent(dataset.INPUT)
        tester.final_evaluate(predictor.model)
        return

    for epoch in range(config.epochs):
        if epoch - validator.best_epoch > 10:
            return

        model.train()
        pbar = tqdm(train_dl)
        total_loss = 0
        n_correct = 0
        cnt = 0
        for batch in pbar:
            batch_size = len(batch.tgt)

            loss, acc = model.loss_n_acc(batch.input, batch.tgt)
            total_loss += loss.item() * batch_size
            cnt += batch_size
            n_correct += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if cnt % (5 * batch_size) == 0:
                pbar.set_description('E{:02d}, loss:{:.4f}, acc:{:.4f}, lr:{}'
                                     .format(epoch,
                                             total_loss / cnt if cnt else 0,
                                             n_correct / cnt if cnt else 0,
                                             optimizer.param_groups[0]['lr']))
                pbar.refresh()

        model.eval()
        validator.evaluate(model, epoch)
        summ = {
            'Eval': '(e{:02d},train)'.format(epoch),
            'loss': total_loss / cnt,
            'acc': n_correct / cnt,
        }
        validator.write_summary(summ=summ)
        validator.write_summary(epoch=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='AttBiLSTM', help="The model to be trained")
    parser.add_argument('--use_gpu', type=int, default=1, help="whether use gpu")
    parser.add_argument('--data_dir', type=str, default='data/', help='directory to data files')
    parser.add_argument('--train_fname', type=str, default='train.csv', help="the filename of training data")
    parser.add_argument('--valid_fname', type=str, default='valid.csv', help="the filename of validation data")
    parser.add_argument('--test_fname', type=str, default='test.csv', help="the filename of test data")
    parser.add_argument('--load_model', type=str, default='', help="path to pretrained model")
    parser.add_argument('--save_dir', type=str, default='tmp/', help='directory to save output files')
    parser.add_argument('--save_log_fname', type=str, default='run_log.txt', help='file name to save training logs')
    parser.add_argument('--save_model_fname', type=str, default='model', help='file to torch.save(model)')
    parser.add_argument('--save_vocab_fname', type=str, default='vocab.json', help='file name to save vocab')

    args = parser.parse_args()
    train(args)

