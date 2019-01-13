# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 12:28:54 2018

@author: truthless
"""

import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from DataManager import GO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HERD(nn.Module):

    def __init__(self, rnn_type, word_vector, nvoc, nembed, nhid,
                 teacher_forcing=0.5, nlayers=1, dropout=0):
        super(HERD, self).__init__()
        
        self.nhid = nhid #H
        self.nvoc = nvoc #C
        self.nembed = nembed #V
        self.teacher_forcing = teacher_forcing
        self.encoder = Encoder(nvoc, nembed, nhid, rnn_cell=rnn_type, variable_lengths=True,
                               embedding=word_vector)
                   
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn_decoder = getattr(nn, rnn_type)(nembed, nhid, nlayers, dropout=dropout)
            self.rnn_context = getattr(nn, rnn_type)(nhid, nhid, nlayers, dropout=dropout)
        else:
            raise ValueError("rnn_type should be GRU or LSTM! {0}".format(rnn_type))
        self.fc_stop_word = nn.Linear(nhid, 2)
        self.text_decoder = nn.Linear(nhid, nvoc)

    def forward(self, x, x_len, y, y_len, use_teacher_forcing=None):
        """
        Parameters
        ----------
        x : (batch, sequence length, utterance length)
        x_len : (batch, sequence length)
        y : (batch, target sequence length)
        y_len : (batch)
        """     
        batch_size = x.shape[0]
        sequence_size = x.shape[1]
        context = Variable(torch.zeros(sequence_size, batch_size, self.nhid)).to(device=device)
        for t in range(sequence_size):
            _, hidden = self.encoder(x[:, t, :], x_len[:, t]) #(batch, max_len, H), (nlayer*ndir, batch, H)
            context[t] = hidden
        _, hidden = self.rnn_context(context) #, (nlayer*ndir, batch, H)
                        
        target_max_len = max(y_len)
        outputs = torch.zeros_like(y).long().to(device=device)
        word_probs = Variable(torch.zeros(batch_size, target_max_len, self.nvoc)).to(device=device) 
        
        if use_teacher_forcing is None:
            use_teacher_forcing = random.random() < self.teacher_forcing
        token_input = Variable(torch.LongTensor([GO] * batch_size)).to(device=device) 
        rnn_input = self.encoder.embedding(token_input) #(batch, V)

        for t in range(target_max_len):
            rnn_input = rnn_input.unsqueeze(0)
            rnn_output, hidden = self.rnn_decoder(rnn_input, hidden) 
            rnn_output = rnn_output.squeeze(0) #(batch, H)
            
            logits = self.text_decoder(rnn_output) #(batch, C)
                            
            word_logits = logits
            results = torch.argmax(word_logits, dim=-1).detach()
            outputs[:, t] = results
                
            word_probs[:, t, :] = word_logits
        
            if use_teacher_forcing:
                rnn_input = self.encoder.embedding(y[:, t]) #(batch, V)
            else:
                rnn_input = self.encoder.embedding(results)
        
        return outputs, word_probs

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, input_dropout_p=0, dropout_p=0, n_layers=1, 
                 rnn_cell='GRU', variable_lengths=False, embedding=None, update_embedding=True):
        super(Encoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell == 'LSTM':
            self.rnn_cell = nn.LSTM
        elif rnn_cell == 'GRU':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))
        
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of
              the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the
              features in the hidden state h
        """
        if self.variable_lengths:
            sort_index = torch.sort(-input_lengths)[1]
            unsort_index = torch.sort(sort_index)[1]
            input_var = input_var[sort_index]
            input_lengths = input_lengths[sort_index]
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.rnn_cell == nn.LSTM:
            hidden = hidden[0]
        if self.variable_lengths:
            hidden = torch.transpose(hidden, 0, 1)[unsort_index]
            hidden = torch.transpose(hidden, 0, 1)
            output, _ = pad_packed_sequence(output, batch_first=True)
            output = output[unsort_index]
        return output, hidden
