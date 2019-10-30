from __future__ import division, print_function
import os
import json
import torch
from utils import fwrite, shell

from model import AttBiLSTM


class Validator:
    def __init__(self, dataloader=None, save_log_fname='run_log.txt',
                 save_model_fname='model.torch', save_dir='tmp/',
                 valid_or_test='valid', vocab_itos=dict(), label_itos=dict(), with_ent=True):
        self.avg_loss = 0
        self.dataloader = dataloader
        self.save_log_fname = os.path.join(save_dir, save_log_fname)
        self.save_model_fname = os.path.join(save_dir, save_model_fname)
        self.valid_or_test = valid_or_test
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.save_dir = save_dir
        self.vocab_itos = vocab_itos
        self.label_itos = label_itos
        self.with_ent = with_ent

    def evaluate(self, model, epoch):
        error = 0
        count = 0
        n_correct = 0

        for batch_ix, batch in enumerate(self.dataloader):
            batch_size = len(batch.tgt)
            if self.with_ent is True:
                loss, acc = model.loss_n_acc(batch.input, batch.tgt, batch.ent1, batch.ent2)
            else:
                loss, acc = model.loss_n_acc(batch.input, batch.tgt)

            error += loss.item() * batch_size
            count += batch_size
            n_correct += acc
        avg_loss = (error / count)
        self.avg_loss = avg_loss
        self.acc = (n_correct / count)

        if (self.valid_or_test == 'valid') and (avg_loss < self.best_loss):
            self.best_loss = avg_loss
            self.best_epoch = epoch

            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'epoch': epoch,
            }
            torch.save(checkpoint, self.save_model_fname)

    def write_summary(self, epoch=0, summ=None):
        def _format_value(v):
            if isinstance(v, float):
                return '{:.4f}'.format(v)
            elif isinstance(v, int):
                return '{:02d}'.format(v)
            else:
                return '{}'.format(v)

        summ = {
            'Eval': '(e{:02d},{})'.format(epoch, self.valid_or_test),
            'loss': self.avg_loss,
            'acc': self.acc,
        } if summ is None else summ
        summ = {k: _format_value(v) for k, v in summ.items()}
        writeout = json.dumps(summ)

        fwrite(writeout + '\n', self.save_log_fname, mode='a')
        printout = '[Info] {}'.format(writeout)
        print(printout)

        return writeout

    def reduce_lr(self, opt):
        if self.avg_loss > self.best_loss:
            for g in opt.param_groups:
                g['lr'] = g['lr'] / 2

    def final_evaluate(self, model,
                       perl_fname='eval/semeval2010_task8_scorer-v1.2.pl'):
        preds = []
        truths = []
        for batch in self.dataloader:
            if self.with_ent is True:
                pred = model.predict(batch.input, batch.ent1, batch.ent2)
            else:
                pred = model.predict(batch.input)

            preds += pred

            truth = batch.tgt.view(-1).detach().cpu().numpy().tolist()
            truths += truth

        pred_fname = os.path.join(self.save_dir, 'tmp_pred.txt')
        truth_fname = os.path.join(self.save_dir, 'tmp_truth.txt')
        result_fname = os.path.join(self.save_dir, 'tmp_result.txt')

        writeout = ["{}\t{}\n".format(ix, self.label_itos[pred]) for ix, pred in
                    enumerate(preds)]
        fwrite(''.join(writeout), pred_fname)

        writeout = ["{}\t{}\n".format(ix, self.label_itos[truth]) for ix, truth
                    in enumerate(truths)]
        fwrite(''.join(writeout), truth_fname)

        cmd = 'perl {} {} {}'.format(perl_fname, pred_fname, truth_fname)
        stdout, _ = shell(cmd, stdout=True)
        fwrite(stdout, result_fname)


if __name__ == '__main__':
    pass
