import argparse
def define_args(filename, description):
    parser = argparse.ArgumentParser('./{}.py'.format(filename), description=description)
    return parser
def add_general_options(parser, **kwargs):
    parser.add_argument('--dataset',default ='MNIST',choices=['MNIST', 'CIFAR10', 'CIFAR100'])