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
    parser.add_argument('--epoch', type=int, default=50, help="Max number of epoch")
    parser.add_argument('--topic', type=int, default=10, help="Number of topics")
    parser.add_argument('--embed', type=int, default=300, help="Dimension of word vector")
    parser.add_argument('--rnn_dim', type=int, default=64, help="Dimension of hidden units of the RNN")
    parser.add_argument('--infer_dim', type=int, default=64, help="Dimension of the inference network hidden layer")
    parser.add_argument('--batch', type=int, default=64, help="Batch size")
    parser.add_argument('--print_per_batch', type=int, default=20, help="Print results every XXX batches")
    parser.add_argument('--load_pretrain', type=bool, default=False, help="whether to load pretrained model")
    parser.add_argument('--load', type=str, default='./checkpoints/model_2019-01-16_10_47_40::DCR_0', help="Directory to load model")
    parser.add_argument('--save', type=str, default='checkpoints', help="Directory to save model")
    parser.add_argument('--test', type=bool, default=False, help="Set to inference")
    parser.add_argument('--data', type=str, default='data', help="Data directory")
    parser.add_argument('--no_wordvec', type=bool, default=False, help="Set to use random word2vec")
    parser.add_argument('--context_len', type=int, default=3, help="Length of dialogue context used in model")
    parser.add_argument('--stopword_freq_lb', type=int, default=1000, help="according to word frequence to add words to stopwords list, this is the frequence lower bound")
    parser.add_argument('--model', type=str, default="gcn_no_activation", help="which model to use: TopicRNN, gcn_no_activation")
    parser.add_argument('--smooth_method', type=str, default="method4", help="which smooth method to use: method4, original_method4, no")
    return parser
