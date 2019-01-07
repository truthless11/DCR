# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 11:56:26 2019

@author: truthless
"""

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

smoothie = SmoothingFunction().method4

def blue(refs, hyps, lens):
    '''
    ref1 = ['a', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
    ref2 = ['quick', 'brown', 'dogs', 'jump', 'over', 'the', 'lazy', 'fox']
    ref = [ref1, ref2]
    hyp = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the']
    '''
    score = 0
    for i, l in enumerate(lens):
        ref, hyp = refs[i, :l].tolist(), hyps[i, :l].tolist()
        score += sentence_bleu([ref], hyp, smoothing_function=smoothie)
    return score

def language_model_p(refs, word_p, lens):
    p = 0
    for i, l in enumerate(lens):
        for j in range(l):
            p += np.log(word_p[i, j, refs[i, j]].item() + 1e-100)
    return p

def perplexity(p, N):
    return np.exp(-1 / N * p)
