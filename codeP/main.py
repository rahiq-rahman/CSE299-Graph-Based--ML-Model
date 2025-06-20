import sys
import os
sys.path.append(os.path.dirname(__file__))

import argparse
from example import exampleFunction

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required = True, type=str)
parser.add_argument('--model', required = True, type=str)
parser.add_argument('--otherOptions', required = False, type=str, default=None)
parser.add_argument('--message', required = False, type=str, default=None, help='Custom Message')

args = parser.parse_args()

exampleFunction(args.message)

