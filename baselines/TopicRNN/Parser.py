# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 12:53:56 2018

@author: truthless
"""

import argparse

def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='DCR', help="Filename of log file")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--teacher', type=float, default=0.5, help="Teacher forcing ratio")
    parser.add_argument('--seed', type=int, default=1, help="Random seed")
    parser.add_argument('--rnn', type=str, default='GRU', help="Rnn cell type")
    parser.add_argument('--epoch', type=int, default=500, help="Max number of epoch")
    parser.add_argument('--topic', type=int, default=20, help="Number of topics")
    parser.add_argument('--embed', type=int, default=300, help="Dimension of word vector")
    parser.add_argument('--rnn_dim', type=int, default=300, help="Dimension of hidden units of the RNN")
    parser.add_argument('--infer_dim', type=int, default=100, help="Dimension of the inference network hidden layer")
    parser.add_argument('--batch', type=int, default=32, help="Batch size")
    parser.add_argument('--print_per_batch', type=int, default=20, help="Print results every XXX batches")
    parser.add_argument('--save_per_batch', type=int, default=10, help="Save model every XXX batches")
    parser.add_argument('--load', type=str, default='', help="Directory to load model")
    parser.add_argument('--save', type=str, default='checkpoints', help="Directory to save model")
    parser.add_argument('--test', type=bool, default=False, help="Set to True to inference")
    parser.add_argument('--data', type=str, default='data', help="Data directory")
    parser.add_argument('--no_wordvec', type=bool, default=False, help="Set to True to use random word2vec")
    return parser
