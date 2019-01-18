# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 11:56:26 2019

@author: truthless
"""

from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import torch.nn.functional as F
from DataManager import EOS

epsilon = 1e-10 # for numerical stability

def method4(p_n, references, hypothesis, hyp_len, *args, **kwargs):
    """
    Smoothing method 4:
    Shorter translations may have inflated precision values due to having
    smaller denominators; therefore, we give them proportionally
    smaller smoothed counts. Instead of scaling to 1/(2^k), Chen and Cherry
    suggests dividing by 1/ln(len(T)), where T is the length of the translation.
    """
    invcnt = 1
    for i, p_i in enumerate(p_n):
        if p_i.numerator == 0 and hyp_len != 0:
            incvnt = invcnt * 5 / np.log(hyp_len + epsilon)
            p_n[i] = 1 / incvnt
    return p_n

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
        score += sentence_bleu([ref], hyp, smoothing_function=method4)
    return score

def language_model_p(refs, word_p, lens):
    p = 0
    probs = F.softmax(word_p, dim=-1)
    for i, l in enumerate(lens):
        for j in range(l):
            p += np.log(probs[i, j, refs[i, j]].item() + epsilon)
    return p

def perplexity(p, N):
    return np.exp(-1 / N * p)

def entity_recall(refs, hyps):
    acc, cnt = 0, 0
    for i, ref in enumerate(refs):
        hyp = hyps[i].tolist()
        if EOS in hyp:
            index = hyp.index(EOS)
            hyp = hyp[:index]
        for ent in ref:
            cnt += 1
            if ent in hyp:
                acc += 1
    return acc, cnt
