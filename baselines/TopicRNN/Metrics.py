# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 11:56:26 2019

@author: truthless
"""

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import torch.nn.functional as F
from DataManager import EOS

smoothie = SmoothingFunction().method4

def bleu(refs, hyps, lens):
    '''
    ref1 = ['a', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
    ref2 = ['quick', 'brown', 'dogs', 'jump', 'over', 'the', 'lazy', 'fox']
    ref = [ref1, ref2]
    hyp = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the']
    '''
    score = 0
    for i, l in enumerate(lens):
        ref = refs[i, :(l-1)].tolist()
        hyp = hyps[i].tolist()
        if EOS in hyp:
            index = hyp.index(EOS)
            hyp = hyp[:index]
        score += sentence_bleu([ref], hyp, smoothing_function=smoothie)
    return score

def language_model_p(refs, word_p, lens):
    p = 0
    probs = F.softmax(word_p, dim=-1)
    for i, l in enumerate(lens):
        for j in range(l):
            p += np.log(probs[i, j, refs[i, j]].item() + 1e-100)
    return p

def perplexity(p, N):
    return np.exp(-1 / N * p)
