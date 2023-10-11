import os
import argparse


class Options:

    def initialize(self, parser):
        parser.add_argument('--train_root', default='/data/wy/datasets/Haze1k/train',help='Training path')
        parser.add_argument('--test_root', default='/home/wy/datasets/Haze1k/Haze1k_thin/dataset/test', help='Testing path')
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--epochs', type=int, default=10000, help='Total number of training')
        parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
        parser.add_argument('--device', type=str, default='0', help='GPU id')
        parser.add_argument('--is_continue', default=False, help='Whether to continue')
        parser.add_argument('--model_path', default='./model/last_model.pth', help='Loading model')
        parser.add_argument('--out_hazy_path', default='./images/input')
        parser.add_argument('--out_gt_path', default='./images/targets')
        parser.add_argument('--out_clear_path', default='./images/clear')
        parser.add_argument('--out_log_path', default='./Log/log_train')
        return parser

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = Options().initialize(parser)
opt, _ = parser.parse_known_args()


if not os.path.exists('./model'):
    os.makedirs('./model')
if not os.path.exists('./Log/max_log'):
    os.makedirs('./Log/max_log')
if not os.path.exists(opt.out_log_path):
    os.makedirs(opt.out_log_path)
if not os.path.exists(opt.out_hazy_path):
    os.makedirs(opt.out_hazy_path)
if not os.path.exists(opt.out_gt_path):
    os.makedirs(opt.out_gt_path)
if not os.path.exists(opt.out_clear_path):
    os.makedirs(opt.out_clear_path)