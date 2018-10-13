import argparse

parser = argparse.ArgumentParser(description='Parzen window, log-likelihood estimator')
parser.add_argument('-p', '--path', help='model path')
parser.add_argument('-s', '--sigma', default = None)
parser.add_argument('-d', '--dataset', choices=['mnist', 'tfd'], default='mnist')
parser.add_argument('-f', '--fold', default = 0, type=int)
parser.add_argument('-v', '--valid', default = False, action='store_true')
parser.add_argument('-n', '--num_samples', default=10000, type=int)
parser.add_argument('-l', '--limit_size', default=1000, type=int)
parser.add_argument('-b', '--batch_size', default=100, type=int)
parser.add_argument('-c', '--cross_val', default=10, type=int,
                        help="Number of cross valiation folds")
parser.add_argument('--sigma_start', default=-1, type=float)
parser.add_argument('--sigma_end', default=0., type=float)
args = parser.parse_args()
