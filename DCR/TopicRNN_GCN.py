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

from layers import GraphConvolution, GraphAttention

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

class TopicRNN_GCN(nn.Module):

    def __init__(self, model, rnn_type, word_vector, nvoc, nvoc_nonstop, nembed, nhid, nhid_infer,
                 ntopic, n_entity, nodes_rep, adj, gcn_dropout=0.5, teacher_forcing=0.5, nlayers=1, dropout=0.05):
        super(TopicRNN_GCN, self).__init__()
        
        self.nhid = nhid #H
        self.nhid_infer = nhid_infer #E
        self.ntopic = ntopic #K
        self.nvoc = nvoc #C
        self.nvoc_nonstop = nvoc_nonstop #C-
        self.nembed = nembed #V
        self.teacher_forcing = teacher_forcing
        self.encoder = Encoder(nvoc, nembed, nhid, rnn_cell=rnn_type, variable_lengths=True, embedding=word_vector)
        
        self.fc = nn.Linear(nvoc_nonstop, nhid_infer)
        self.fc_mu = nn.Linear(nhid_infer, nhid_infer)
        self.fc_sigma = nn.Linear(nhid_infer, nhid_infer)
        self.fc_theta = nn.Linear(nhid_infer, ntopic)
        # just use a fully-connected layer to replace gcn
        self.fc_replace_gcn = nn.Linear(nembed, nhid)
        # convert the hidden state to generate a query vector to query at the KG
        self.fc_query = nn.Linear(nhid, nhid)
           
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn_decoder = getattr(nn, rnn_type)(nembed, nhid, nlayers, dropout=dropout)
            self.rnn_context = getattr(nn, rnn_type)(nhid, nhid, nlayers, dropout=dropout)
        else:
            raise ValueError("rnn_type should be GRU or LSTM! {0}".format(rnn_type))
        self.fc_stop_word = nn.Linear(nhid, 2)
        self.text_decoder = nn.Linear(nhid, nvoc)
        self.topic_decoder = nn.Linear(ntopic, nvoc)

        self.n_entity = n_entity
        self.adj = adj.cuda()
        self.nodes_rep = torch.LongTensor(nodes_rep).cuda()
        init = "uniform"
        self.gc1 = GraphConvolution(nembed, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nhid, init=init)
        self.gcn_dropout = gcn_dropout
        self.gcn_normalize = False

        self.model = model


    def get_nodes_embed(self):
        # note self.nodes_rep[0] corresponds to the node "none"
        nodes_embed = self.encoder.embedding(self.nodes_rep[1:]) # [node_number, max_len, embed_dim]: [314, 50, nembed]
        nodes_embed = torch.max(nodes_embed, dim=1)[0] # [314, nembed]
        #nodes_embed = torch.mean(nodes_embed, dim=1) # [314, nembed]

        mode = "gcn_no_activation" # "gcn", "no_gcn", "gcn_no_activation", "gcn_no_dropout", "gcn_no_drop_no_acti"
        mode = self.model
        
        if mode == "gcn":
            nodes_embed = F.leaky_relu(self.gc1(nodes_embed, self.adj)) # [314, nhid]
            nodes_embed = F.dropout(nodes_embed, self.gcn_dropout)
            nodes_embed = F.leaky_relu(self.gc2(nodes_embed, self.adj)) # [314, nhid]
        if mode == "no_gcn":
            nodes_embed = self.fc_replace_gcn(nodes_embed) 
        if mode == "gcn_no_activation":
            nodes_embed = self.gc1(nodes_embed, self.adj) # [314, nhid]
            nodes_embed = F.dropout(nodes_embed, self.gcn_dropout)
            nodes_embed = self.gc2(nodes_embed, self.adj) # [314, nhid]
        if mode == "gcn_no_dropout":
            nodes_embed = F.leaky_relu(self.gc1(nodes_embed, self.adj)) # [314, nhid]
            nodes_embed = F.leaky_relu(self.gc2(nodes_embed, self.adj)) # [314, nhid]
        if mode == "gcn_no_drop_no_acti":
            nodes_embed = self.gc1(nodes_embed, self.adj) # [314, nhid]
            nodes_embed = self.gc2(nodes_embed, self.adj) # [314, nhid]
        #if self.gcn_normalize:
        #    nodes_embed = F.normalize(nodes_embed, p=2, dim=1)

        return nodes_embed


    def forward(self, x, x_len, y, y_len, x_tf, use_teacher_forcing=None):
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
        sequence_size = x.shape[1]
        context = Variable(torch.zeros(sequence_size, batch_size, self.nhid)).to(device=device)
        for t in range(sequence_size):
            _, hidden = self.encoder(x[:, t, :], x_len[:, t]) #(batch, max_len, H), (nlayer*ndir, batch, H)
            context[t] = hidden
        _, hidden = self.rnn_context(context) #, (nlayer*ndir, batch, H)
        
        mu, log_sigma = self.encode(x_tf) #(batch, E), (batch, E)
        # Compute noisy topic proportions given Gaussian parameters.
        Z = self.reparameterize(mu, log_sigma) #(batch, E)
        theta = F.softmax(self.fc_theta(Z), dim=1) #(batch, K)
                
        target_max_len = max(y_len)
        outputs = torch.zeros_like(y).long().to(device=device)
        word_probs = Variable(torch.zeros(batch_size, target_max_len, self.nvoc)).to(device=device) 
        indicator_probs = Variable(torch.zeros(batch_size, target_max_len, 2)).to(device=device) 
        entity_probs = Variable(torch.zeros(batch_size, target_max_len, self.n_entity)).to(device=device) 
        
        if use_teacher_forcing is None:
            use_teacher_forcing = random.random() < self.teacher_forcing
        token_input = Variable(torch.LongTensor([GO] * batch_size)).to(device=device) 
        rnn_input = self.encoder.embedding(token_input) #(batch, V)

        for t in range(target_max_len):
            rnn_input = rnn_input.unsqueeze(0)
            rnn_output, hidden = self.rnn_decoder(rnn_input, hidden) 
            rnn_output = rnn_output.squeeze(0) #(batch, H)
            
            logits = self.text_decoder(rnn_output) #(batch, C)

                
            stopword_logits = torch.sigmoid(self.fc_stop_word(rnn_output)) #(batch, 2)
            stopword_predictions = torch.argmax(stopword_logits, dim=-1).unsqueeze(-1)
            
            topic_additions = self.topic_decoder(theta) #(batch, C)            
            topic_additions[:, :4] = 0  # Padding & Unknowns will be treated as stops. 
            topic_mask = (1 - stopword_predictions).expand(-1, self.nvoc)
            topic_additions = topic_additions * topic_mask.float()
            
            word_logits = logits + topic_additions

            results = torch.argmax(word_logits, dim=-1).detach()
            outputs[:, t] = results

            word_probs[:, t, :] = word_logits 
            indicator_probs[:, t, :] = stopword_logits

            if self.model != "TopicRNN":
                # start: entity selection 
                nodes_embed = self.get_nodes_embed() # [314, nhid] 
                zero_node = torch.zeros([1, nodes_embed.shape[1]]).cuda()
                nodes_embed = torch.cat([zero_node, nodes_embed], dim=0) # [315, nhid]

                #query = self.fc_query(hidden)
                query = hidden
                tmp_hidden = query.squeeze(0).unsqueeze(1).expand(-1, nodes_embed.shape[0], -1) # [batch_size, 315, nhid]
                nodes_embed = nodes_embed.unsqueeze(0).expand(tmp_hidden.shape) # [batch_size, 315, nhid]
                entity_prob = torch.sum(tmp_hidden * nodes_embed, dim=2) # [batch_size, 315]
                entity_probs[:, t, :] = entity_prob
                # end: entity selection 

            if use_teacher_forcing:
                rnn_input = self.encoder.embedding(y[:, t]) #(batch, V)
            else:
                rnn_input = self.encoder.embedding(results)
        
        return outputs, word_probs, indicator_probs, mu, log_sigma, entity_probs
    
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
