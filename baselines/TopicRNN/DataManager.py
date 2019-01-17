# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:14:04 2018

@author: truthless
"""

import random
from stop import STOP_WORDS
import numpy as np
import torch
import torch.utils.data as data

PAD = 0
UNK = 1 #OOV
GO = 2
EOS = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataManager:
    
    def __init__(self, path, no_pretrain_word2vec, dim):
               
        #read text
        self.text = {}
        for name in ["train", "valid", "test"]:
            self.text[name] = []
            with open("{0}/{1}.txt".format(path, name)) as fl:
                for line in fl:
                    self.text[name].append(line.strip().lower().split('\t'))

        #arrange words
        wordscount = {}
        for name in ["train", "valid"]:
            texts = self.text[name]
            for item in texts:
                words = item[0].split() + item[1].split()
                for word in words:
                    if word in wordscount:
                        wordscount[word] += 1
                    else:
                        wordscount[word] = 1
        wordssorted = sorted(wordscount.items(), key = lambda d: (d[1],d[0]), reverse=True) 
        self.word2index = {'<PAD>':0, '<UNK>':1, '<GO>':2, '<EOS>':3}
        for i, (key, value) in enumerate(wordssorted):
            if value == 20:
                break
            self.word2index[key] = i + 4 #PAD,UNK,GO,EOS
        print("Voc size {0}".format(len(self.word2index)))
        self.stop_words_index = set([PAD, UNK, GO, EOS])
        self.stop_words_index |= set([self.word2index[word] for word in STOP_WORDS])
        self.index2word = dict((v, k) for k, v in self.word2index.items())
        
        #load word vector
        if no_pretrain_word2vec:
            self.vector = None
        else:
            self.vector = 0.1 * np.random.rand(len(self.word2index), dim)
            with open("{0}/vector.txt".format(path)) as fl:
                for line in fl:
                    vec = line.strip().split()
                    word = vec[0].lower()
                    vec = list(map(float, vec[1:]))
                    if word in self.word2index:
                        self.vector[self.word2index[word]] = np.asarray(vec)
            self.vector = torch.Tensor(self.vector)
        
        # compute tf
        len_voc = len(self.word2index.values())
        self.index2nonstop = {}
        cnt = 0
        for i in range(len_voc):
            if i not in self.stop_words_index:
                self.index2nonstop[i] = cnt
                cnt += 1
    
        #get index
        self.data = {}
        for name in ["train", "valid", "test"]:
            self.data[name] = []
            for item in self.text[name]:
                indices = [[],[]]
                for i in [0, 1]:
                    words = item[i].split()
                    indices[i] = [self.word2index[word] if word in self.word2index 
                                  else UNK for word in words]
                    if i == 1: # answer from system
                        indices[1].append(EOS)
                nonstop_indices = [[self.index2nonstop[index] for index in indices[i] 
                                    if index in self.index2nonstop] for i in [0, 1]]
                tf = [torch.zeros(cnt), torch.zeros(cnt)]
                for i in [0, 1]:
                    normal = len(nonstop_indices[i])
                    for j in nonstop_indices[i]:
                        tf[i][j] += 1. / normal
                self.data[name].append(indices)
           
    def create_dataset(self, name, batch_size):
        datas = self.data[name]
        src_seqs, trg_seqs = [], []
        trg_stops, src_tfs = [], []
        nonstop_voc_size = len(self.index2nonstop)
        for item in datas:
            src, trg = item
            tensor_src, tensor_trg = torch.LongTensor(src), torch.LongTensor(trg)
            src_seqs.append(tensor_src)
            trg_seqs.append(tensor_trg)
            
            trg_stop = torch.zeros_like(tensor_trg)
            for i, index in enumerate(trg):
                if index in self.stop_words_index:
                    trg_stop[i] = 1
            trg_stops.append(trg_stop)
            
            src_tf = torch.zeros(nonstop_voc_size)
            for i, index in enumerate(src):
                if index not in self.stop_words_index:
                    src_tf[self.index2nonstop[index]] += 1
            if src_tf.sum().item() > 0:
                src_tf /= src_tf.sum()
            src_tfs.append(src_tf)
            
        dataset = Dataset(src_seqs, trg_seqs, trg_stops, src_tfs)
        dataloader = data.DataLoader(dataset, batch_size, True, collate_fn=pad_packed_collate)
        return dataloader
    
    def interpret(self, preds, refs, lens, f):
        i = random.randrange(0, len(lens))
        l = max(lens)
        for j in range(l):
            word = self.index2word[preds[i][j].item()]
            print(word, end=' ')
            f.write('{0} '.format(word))
            if word == '<EOS>':
                break
        print()
        f.write('\n')
        
        l = lens[i]
        for j in range(l):
            word = self.index2word[refs[i][j].item()] 
            print(word, end=' ')
            f.write('{0} '.format(word))
        print()
        f.write('\n')

class Dataset(data.Dataset):
    
    def __init__(self, src_seqs, trg_seqs, trg_stops, src_tfs):
        self.src_seqs = src_seqs
        self.trg_seqs = trg_seqs
        self.trg_stops = trg_stops
        self.src_tfs = src_tfs
        self.num_total_seqs = len(src_seqs)
        
    def __getitem__(self, index):
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        trg_stop = self.trg_stops[index]
        src_tf = self.src_tfs[index]
        return src_seq, trg_seq, trg_stop, src_tf
    
    def __len__(self):
        return self.num_total_seqs
        
def pad_packed_collate(batch_data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        return padded_seqs, lengths
    
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    batch_data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences    
    src_seqs, trg_seqs, trg_stops, src_tfs = zip(*batch_data)
    src_seqs, src_lens = merge(src_seqs)
    trg_seqs, trg_lens = merge(trg_seqs)
    trg_stops, _ = merge(trg_stops)
    return (src_seqs.to(device=device), torch.LongTensor(src_lens).to(device=device), 
            trg_seqs.to(device=device), trg_lens, 
            trg_stops.to(device=device), torch.stack(src_tfs).to(device=device))
    