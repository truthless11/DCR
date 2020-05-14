# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:02:16 2018

@author: truthless
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_function(word_probs, y, indicator_probs, y_stop, mu, logvar, lens, entity_p, entity_grd, entity_mask):
    max_len = max(lens)
    expand_len = [([1]*l + [0]*(max_len-l)) for l in lens]
    mask = Variable(torch.Tensor(expand_len)).to(device=device)
    CE = sequence_cross_entropy_with_logits(word_probs, y, mask)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    SCE = sequence_cross_entropy_with_logits(indicator_probs, y_stop, mask)
    ENTCE = sequence_cross_entropy_with_logits(entity_p, entity_grd, entity_mask)
    #ENTCE = sequence_cross_entropy_with_logits(entity_p, entity_grd, mask)
    return CE, KLD, SCE, ENTCE

def sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                       targets: torch.LongTensor,
                                       weights: torch.FloatTensor,
                                       batch_average: bool = True) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.

    Parameters
    ----------
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch, sequence_length)
    batch_average : bool, optional, (default = True).
        A bool indicating whether the loss should be averaged across the batch,
        or returned as a vector of losses per batch element.

    Returns
    -------
    A torch.FloatTensor representing the cross entropy loss.
    If ``batch_average == True``, the returned loss is a scalar.
    If ``batch_average == False``, the returned loss is a vector of shape (batch_size,).

    """
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    # Contribution to the negative log likelihood only comes from the exact indices
    # of the targets, as the target distributions are one-hot. Here we use torch.gather
    # to extract the indices of the num_classes dimension which contribute to the loss.
    # shape : (batch * sequence_length, 1)
    negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()
    # shape : (batch_size,)
    per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)

    if batch_average:
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    return per_batch_loss 
