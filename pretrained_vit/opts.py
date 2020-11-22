import argparse
import os, sys

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--exp_id', default='default')

        
        self.parser.add_argument('--gpus', default='1')
        self.parser.add_argument('--num_workers', type=int, default=8)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--num_iters', type=int, default=-1)
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--num_epochs', type=int, default=100)
        
    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.root_dir = os.path.join(os.path.dirname(__file__), '.')
        opt.data_path = os.path.join(opt.root_dir, '../data/imagenet')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp')
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)

        return opt
