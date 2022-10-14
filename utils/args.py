import argparse
import os

def dir_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        type=dir_path,
        dest='config',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args
