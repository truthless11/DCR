# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 12:53:56 2018

@author: truthless
"""

import argparse

def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, default='HRL', help="Filename of log file")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--teacher', type=float, default=0.5, help="Teacher forcing ratio")
    parser.add_argument('--seed', type=int, default=1, help="Random seed")
    parser.add_argument('--rnn', type=str, default='GRU', help="Rnn cell type")
    parser.add_argument('--epoch', type=int, default=15, help="Max number of epoch")
    parser.add_argument('--topic', type=int, default=10, help="Number of topics")
    parser.add_argument('--embed', type=int, default=100, help="Dimension of word vector")
    parser.add_argument('--rnn_dim', type=int, default=50, help="Dimension of hidden units of the RNN")
    parser.add_argument('--infer_dim', type=int, default=30, help="Dimension of the inference network hidden layer")
    parser.add_argument('--batch', type=int, default=32, help="Batch size on training")
    parser.add_argument('--print_per_batch', type=int, default=20, help="Print results every XXX batches")
    parser.add_argument('--start', type=str, default='', help="Directory to load model")
    parser.add_argument('--test', type=bool, default=False, help="Set to True to inference")
    parser.add_argument('--datapath', type=str, default='./data/', help="Data directory")
    return parser