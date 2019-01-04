# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 12:28:54 2018

@author: truthless
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from DataManager import GO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TopicRNN(nn.Module):

    def __init__(self, rnn_type, word_vector, nvoc, nvoc_nonstop, nembed, nhid, nhid_infer,
                 ntopic, teacher_forcing=0.5, nlayers=1, dropout=0):
        super(TopicRNN, self).__init__()
        
        self.nhid = nhid #H
        self.nhid_infer = nhid_infer #E
        self.ntopic = ntopic #K
        self.nvoc = nvoc #C
        self.nvoc_nonstop = nvoc_nonstop #C-
        self.nembed = nembed #V
        self.teacher_forcing = teacher_forcing
        self.encoder = Encoder(nvoc, nembed, nhid, rnn_cell=rnn_type, variable_lengths=True,
                               embedding=word_vector)
        
        self.fc = nn.Linear(nvoc_nonstop, nhid_infer)
        self.fc_mu = nn.Linear(nhid_infer, nhid_infer)
        self.fc_sigma = nn.Linear(nhid_infer, nhid_infer)
        self.fc_theta = nn.Linear(nhid_infer, ntopic)
           
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(nembed, nhid, nlayers, dropout=dropout)
        else:
            raise ValueError("rnn_type should be GRU or LSTM! {0}".format(rnn_type))
        self.fc_stop_word = nn.Linear(nhid, 2)
        self.text_decoder = nn.Linear(nhid, nvoc)
        self.topic_decoder = nn.Linear(ntopic, nvoc)

    def forward(self, x, x_len, x_stop, x_tf, y, y_len, training=True, use_teacher_forcing='random'):
        """
        Parameters
        ----------
        x : (batch, sequence length)
        x_len : (batch)
        x_stop : (batch, sequence length)
        x_tf: (batch, nonstop vocabulary size)
        y : (batch, target sequence length)
        y_len : (batch)
        """     
        batch_size = x.shape[0]
        context, hidden = self.encoder(x, x_len) #(batch, max_len, H), (nlayer*ndir, batch, H)
        
        mu, log_sigma = self.encode(x_tf) #(batch, E), (batch, E)
        # Compute noisy topic proportions given Gaussian parameters.
        Z = self.reparameterize(mu, log_sigma) #(batch, E)
        theta = F.softmax(self.fc_theta(Z), dim=1) #(batch, K)
                
        target_max_len = max(y_len)
        outputs = torch.zeros_like(y).long().to(device=device)
        word_probs = Variable(torch.zeros(batch_size, target_max_len, self.nvoc)).to(device=device) 
        indicator_probs = Variable(torch.zeros(batch_size, target_max_len, 2)).to(device=device) 
        
        if use_teacher_forcing == 'random':
            use_teacher_forcing = random.random() < self.teacher_forcing
        token_input = Variable(torch.LongTensor([GO] * batch_size)).to(device=device) 
        rnn_input = self.encoder.embedding(token_input) #(batch, V)

        for t in range(target_max_len):
            rnn_input = rnn_input.unsqueeze(0)
            rnn_output, hidden = self.rnn(rnn_input, hidden) 
            rnn_output = rnn_output.squeeze(0) #(batch, H)
            
            logits = self.text_decoder(rnn_output) #(batch, C)
                
            stopword_logits = torch.sigmoid(self.fc_stop_word(rnn_output)) #(batch, 2)
            if training:
                stopword_predictions = torch.multinomial(stopword_logits, 1) #(batch)
            else:
                stopword_predictions = torch.argmax(stopword_logits, dim=-1).unsqueeze(-1)
            
            topic_additions = self.topic_decoder(theta) #(batch, C)            
            topic_additions[:, :2] = 0  # Padding & Unknowns will be treated as stops.
            topic_mask = (1 - stopword_predictions).expand(-1, self.nvoc)
            topic_additions = topic_additions * topic_mask.float()
            
            probs = F.softmax(logits + topic_additions, dim=1)
            if training:
                outputs[:, t] = torch.multinomial(probs, 1).squeeze(1)
            else:
                outputs[:, t] = torch.argmax(probs, dim=-1)
                
            word_probs[:, t, :] = probs
            indicator_probs[:, t, :] = stopword_logits
        
            if use_teacher_forcing:
                rnn_input = self.encoder.embedding(y[:, t].clone()) #(batch, V)
            else:
                rnn_input = self.encoder.embedding(outputs[:, t].clone())
        
        return outputs, word_probs, indicator_probs, mu, log_sigma
    
    def encode(self, x):
        h = F.relu(self.fc(x)) #(batch, E)
        return self.fc_mu(h), self.fc_sigma(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(device=device)
        return eps.mul(std).add_(mu)

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
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = pad_packed_sequence(output, batch_first=True)
        return output, hidden
