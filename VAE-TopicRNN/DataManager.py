# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:14:04 2018

@author: truthless
"""

import re
from stop import STOP_WORDS
import torch
import torch.utils.data as data

PAD = 0
UNK = 1 #OOV
GO = 2
EOS = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataManager:
    
    def __init__(self, path):
               
        #read text
        self.text = {}
        for name in ["train", "valid", "test"]:
            self.text[name] = []
            with open(path+(name+".txt")) as fl:
                lines = fl.readlines()
                for line in lines:
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
        self.word2index = {}
        punctuation = [EOS]
        for i, (key, value) in enumerate(wordssorted):
            if value == 1:
                break
            self.word2index[key] = i + 4 #PAD,UNK,GO,EOS
            if not re.search(r'\w', key):
                punctuation.append(key)
        self.stop_words_index = set(punctuation)
        self.stop_words_index |= set([self.word2index[word] for word in STOP_WORDS if word in self.word2index])
    
        #get index
        self.data = {}
        for name in ["train", "valid", "test"]:
            self.data[name] = []
            for item in self.text[name]:
                indices = [[],[]]
                for i in [0, 1]:
                    words = item[i].split()
                    indices[i] = [self.word2index[word] if word in self.word2index else UNK for word in words]
                    if i == 1: # answer from system
                        indices[1].append(EOS)
                self.data[name].append(indices)
            assert (len(self.data[name]) == len(self.text[name]))
                
    
    def compute_stopword(self, y):
        res = torch.zeros_like(y).to(device=device)
        for i, row in enumerate(y):
            words_index = row.tolist()
            res[i] = torch.LongTensor([int(index in self.stop_words_index) for index in words_index])
        return res

class Dataset(data.Dataset):
    
    def __init__(self, src_seqs, trg_seqs, src_stops, trg_stops):
        self.src_seqs = src_seqs
        self.trg_seqs = trg_seqs
        self.src_stops = src_stops
        self.trg_stops = trg_stops
        self.num_total_seqs = len(src_seqs)
        
    def __getitem__(self, index):
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        src_stop = self.src_stops[index]
        trg_stop = self.trg_stops[index]
        return src_seq, trg_seq, src_stop, trg_stop
    
    def __len__(self):
        return self.num_total_seqs
        
def create_dataset(datas, stop_words, batch_size):
    src_seqs, trg_seqs = [], []
    src_stops, trg_stops = [], []
    for item in datas:
        src, trg = item
        tensor_src, tensor_trg = torch.LongTensor(src), torch.LongTensor(trg)
        src_seqs.append(tensor_src)
        trg_seqs.append(tensor_trg)
        src_stop, trg_stop = torch.zeros_like(tensor_src), torch.zeros_like(tensor_trg)
        for i in range(len(src)):
            if src[i] in stop_words:
                src_stop[i] = 1
        for i in range(len(trg)):
            if trg[i] in stop_words:
                trg_stop[i] = 1
        src_stops.append(src_stop)
        trg_stops.append(trg_stop)
    dataset = Dataset(src_seqs, trg_seqs, src_stops, trg_stops)
    dataloader = data.DataLoader(dataset, batch_size, True, collate_fn=pad_packed_collate)
    return dataloader

def pad_packed_collate(batch_data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths
    
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    batch_data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences    
    src_seqs, trg_seqs, src_stops, trg_stops = zip(*batch_data)
    src_seqs, src_lens = merge(src_seqs)
    src_stops, _ = merge(src_stops)
    trg_seqs, trg_lens = merge(trg_seqs)
    trg_stops, _ = merge(trg_stops)
    results = [src_seqs, src_lens, src_stops, trg_seqs, trg_lens, trg_stops]
    for i in range(4):
        results[i] = torch.LongTensor(results[i]).to(device=device)
    return results
    