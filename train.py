import argparse
from model import AttentionBasedBiLSTM
import load_data as ld

def train(args):




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='AttBiLSTM', help="The model to be trained")
    parser.add_argument('--use_gpu', type=int, default=1, help="whether use gpu")

    args = parser.parse_args()
    train(args)