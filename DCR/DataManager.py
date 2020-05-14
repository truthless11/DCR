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

import re
import json
import networkx as nx
import scipy.sparse as sp

PAD = 0
UNK = 1 #OOV
GO = 2
EOS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataManager:
    
    def __init__(self, stopword_freq_lb, path, no_pretrain_word2vec, dim, context_len):
               
        #read text
        self.text = {}
        for name in ["train", "valid", "test"]:
            self.text[name] = []
            entities = []
            file_path = "{0}/{1}_ent_1.txt".format(path, name)
            for line in open(file_path):
                entities.append(line.strip())

            sys_ans_utt_ori = []
            file_path = "{0}/{1}_ans_utt_ori_1.txt".format(path, name)
            for line in open(file_path):
                sys_ans_utt_ori.append(line.strip())

            cnt = 0
            file_path = "{0}/{1}_utt_1.txt".format(path, name)
            for line in open(file_path):
                utterances = line.strip().split('\t')
                utterances = [''] * (context_len - len(utterances)) + utterances
                self.text[name].append([utterances[-context_len:-1], utterances[-1], entities[cnt], sys_ans_utt_ori[cnt]])
                cnt += 1

        #arrange words
        wordscount = {}
        for name in ["train", "valid"]:
            texts = self.text[name]
            for item in texts:
                words = item[0][-1].split() + item[1].split()
                for word in words:
                    if word in wordscount:
                        wordscount[word] += 1
                    else:
                        wordscount[word] = 1

        wordssorted = sorted(wordscount.items(), key = lambda d: (d[1],d[0]), reverse=True) 
        
        output = open("word_cnt_stat.txt", "w")
        for i, (key, value) in enumerate(wordssorted):
            output.write(str(value) + ":" + str(key) + "\n")

        self.word2index = {'<PAD>':0, '<UNK>':1, '<GO>':2, '<EOS>':3}
        stopwords_self = set()
        for i, (key, value) in enumerate(wordssorted):
            if value <= 5:
                break
            self.word2index[key] = i + 4 #PAD,UNK,GO,EOS
            if value >= stopword_freq_lb:
                stopwords_self.add(key)

        # to add all entity name into vocab
        entity_list = json.load(open("./data/entity_list_simple.json"))
        start_idx = len(self.word2index)
        for entity_name in entity_list:
            entity_name = entity_name.split("::")[-1]
            if entity_name not in self.word2index:
                self.word2index[entity_name] = start_idx
                start_idx += 1

        self.stop_words_index = set([PAD, UNK, GO, EOS])
        #self.stop_words_index |= set([self.word2index[word] for word in STOP_WORDS 
        #                              if word in self.word2index])
        # here we add all words into stopword list
        self.stop_words_index |= set([self.word2index[word] for word in stopwords_self])

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
    
        # for graph initialization
        self.node_id_map, self.id_nodename_map = self.get_node_id_map()
        self.node_info_map, self.nodename_attr_map = self.get_node_info()
        self.adj = self.get_adj_mat("./data/adj_simple.json", self.node_id_map)
        self.nodes_rep = self.get_nodes_rep(self.node_id_map, self.node_info_map)
        self.n_entity = len(self.node_id_map)

        #get index
        self.data = {}
        for name in ["train", "valid", "test"]:
            self.data[name] = []
            for number, item in enumerate(self.text[name]):
                len_u = len(item[0])
                indices = [[], [[] for _ in range(len_u)], [], [], [], []] #src_len, src, trg, trg_entities, trg_entities_mask
                indices[0] = [u.count(' ')+1 for u in item[0]] # on purpose
                max_u_len = max(indices[0])
                # history
                for i in range(len_u):
                    words = item[0][i].split()
                    indices[1][i] = [self.word2index[word] if word in self.word2index 
                                  else UNK for word in words] + [PAD] * (max_u_len - len(words))
                # answer
                words = item[1].split()
                #print("item1:: ", len(words))
                indices[2] = [self.word2index[word] if word in self.word2index 
                              else UNK for word in words]
                indices[2].append(EOS)

                # answer entity
                entities = item[2].split()
                #print("item2 entities:: ", len(entities))
                indices[3] = [self.node_id_map[entity_name] for entity_name in entities]
                indices[3].append(0)

                indices[4] = []
                for x in indices[3]:
                    if x == 0:
                        indices[4].append(0)
                    else:
                        indices[4].append(1)

                # ansuer original sentence
                words = item[3].split()
                indices[5] = words
                indices[5].append("<EOS>")

                if len(indices[2]) != len(indices[3]):
                    print(number, len(indices[2]), len(indices[3]))
                    print(item[1])
                    print(item[2])
                    exit()

                self.data[name].append(indices)


    def get_node_info(self):
        node_info_map = json.load(open("./data/entity_info.json"))
        nodename_attr_map = {}
        for node, res in node_info_map.items():
            node_name = node.split("::")[-1]
            nodename_attr_map[node_name] = res
        return node_info_map, nodename_attr_map


    def post_process(self, outputs, pred_ents, topK=1):
        outputs = outputs.cpu().numpy().tolist()
        pred_ents = pred_ents.cpu().numpy()

        entity_attr_list = {
            "[attraction_address]",
            "[restaurant_address]",
            "[attraction_phone]",
            "[restaurant_phone]",
            "[hotel_address]",
            "[restaurant_postcode]",
            "[attraction_postcode]",
            "[hotel_phone]",
            "[hotel_postcode]",
            "[hospital_phone]"
        }

        lens_new = []
        for i, out in enumerate(outputs):
            new_out = []
            for j, each in enumerate(out):
                if self.index2word[each] == "<$>":
                    pred_ent = np.argmax(pred_ents[i][j])
                    nodename = self.id_nodename_map[pred_ent]
                    new_out.append(nodename)
                elif self.index2word[each] in entity_attr_list:
                    attr_name = self.index2word[each]
                    cnt = 0
                    suc_flag = False
                    for idx, prob in sorted(enumerate(pred_ents[i][j]), key=lambda i: i[1], reverse=True):
                        if suc_flag or cnt >= topK:
                            break
                        nodename = self.id_nodename_map[idx]
                        if nodename not in self.nodename_attr_map:
                            cnt += 1
                            continue
                        for attr, val in self.nodename_attr_map[nodename].items():
                            if attr in attr_name:
                                new_out.append(val)
                                suc_flag = True
                                break
                        cnt += 1
                    if not suc_flag:
                        new_out.append("<UNK>")
                else:
                    new_out.append(self.index2word[each])
                """
                if each == self.word2index["<$>"]:
                    pred_ent = np.argmax(pred_ents[i][j])
                    nodename = self.id_nodename_map[pred_ent]
                    nodename_wordids = [self.word2index[x] for x in nodename.split()]
                    new_out += nodename_wordids
                else:
                    new_out.append(each)
                """
            outputs[i] = new_out

        return outputs 


    def get_nodes_rep(self, node_id_map, node_info_map, max_len=50):
        nodes_rep = []
        nodes_rep_map = []
        for name, id_ in sorted(node_id_map.items(), key=lambda i: i[1]):
            if name == "none" and id_ == 0:
                nodes_rep.append([PAD] * max_len)
                nodes_rep_map.append({"words": ["none"], "idx": [0]})
                continue

            # the attributes used to build relationship
            # attributes as nodes: {"pricerange", "area", "food"}
            # attributes only as relation: {"internet", "parking", "stars", "attraction_type", "hotel_type"}

            # only user node name as node's feature
            name = name.split("::")[-1]
            node_desc = [name]
            nodes_rep_idx = [PAD] * max_len
            nodes_rep_idx[0] = self.word2index[name]
            nodes_rep_word = [name]

            """
            for attr, val in node_info_map.items():
                #if attr in {"address", "area", "pricerange", "introduction", "food", "stars"} or "type" in attr:
                if attr == "introduction":
                    node_desc.append(val)
            node_desc = " ".join(node_desc)
            
            nodes_rep_idx = []
            nodes_rep_word = []
            for each_word in node_desc.split():
                for word in re.split(r'[\[\](::)_]', each_word):
                    if word == "":
                        continue
                    else:
                        if word not in self.word2index:
                            continue
                        else:
                            word_idx = self.word2index[word]
                            nodes_rep_idx.append(word_idx)
                            nodes_rep_word.append(word)
            len_ = len(nodes_rep_idx)
            if len_ >= max_len:
                nodes_rep_idx = nodes_rep_idx[0:max_len]
                nodes_rep_word = nodes_rep_word[0:max_len]
            else:
                nodes_rep_idx += [PAD] * (max_len - len_)
            """

            nodes_rep.append(nodes_rep_idx)
            nodes_rep_map.append({"words": nodes_rep_word, "idx": nodes_rep_idx})

        json.dump(nodes_rep_map, open("nodes_rep_words_idx.json", "w"))
        json.dump(self.word2index, open("word2index.json", "w"))
        #exit()

        return nodes_rep


    def get_node_id_map(self):
        data = json.load(open("./data/entity_list_simple.json"))
        node_id_map = {}
        id_nodename_map = {}
        for i, node in enumerate(data):
            node_id_map[node] = i + 1
            tmp = node.split("::")
            #node_name = " ".join(tmp[1].split("_"))
            node_name = tmp[1]
            id_nodename_map[i+1] = node_name
        node_id_map["none"] = 0
        id_nodename_map[0] = ""

        return node_id_map, id_nodename_map


    def get_adj_mat(self, input_file, item_id_map):
        adj = json.load(open(input_file))
        new_adj = {}
        for i, neibors in adj.items():
            i_idx = item_id_map[i]
            new_adj[i_idx] = []
            for j in neibors:
                j_idx = item_id_map[j]
                new_adj[i_idx].append(j_idx)

        new_adj = nx.adjacency_matrix(nx.from_dict_of_lists(new_adj))
        new_adj = self.normalize_adj(new_adj + sp.eye(new_adj.shape[0]))
        new_adj = torch.FloatTensor(np.array(new_adj.todense()))

        return new_adj


    def normalize_adj(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()

           
    def create_dataset(self, name, batch_size):
        datas = self.data[name]
        src_seq_lens = []
        src_seqs, trg_seqs = [], []
        trg_stops, src_tfs = [], []
        trg_ents, trg_ents_mask, trg_seqs_ori = [], [], []

        nonstop_voc_size = len(self.index2nonstop)
        for item in datas:
            src_len, src, trg, ents, ents_mask, trg_ori = item
            tensor_src_len, tensor_src, tensor_trg = torch.LongTensor(src_len), \
                                                     torch.LongTensor(src), torch.LongTensor(trg)
            src_seq_lens.append(tensor_src_len)
            src_seqs.append(tensor_src)
            trg_seqs.append(tensor_trg)
            
            trg_stop = torch.zeros_like(tensor_trg)
            for i, index in enumerate(trg):
                if index in self.stop_words_index:
                    trg_stop[i] = 1
            trg_stops.append(trg_stop)

            src_tf = torch.zeros(nonstop_voc_size)
            for j, uttr in enumerate(src):
                for i, index in enumerate(uttr):
                    if i == src_len[j]:
                        break
                    if index not in self.stop_words_index:
                        src_tf[self.index2nonstop[index]] += 1
            if src_tf.sum().item() > 0:
                src_tf /= src_tf.sum()
            src_tfs.append(src_tf)

            trg_ents.append(torch.LongTensor(ents))
            trg_ents_mask.append(torch.LongTensor(ents_mask))
            trg_seqs_ori.append(trg_ori)

        print(len(trg_stops), len(trg_seqs), len(trg_ents), len(trg_seqs_ori))

        dataset = Dataset(src_seq_lens, src_seqs, trg_seqs, trg_stops, src_tfs, trg_ents, trg_ents_mask, trg_seqs_ori)
        dataloader = data.DataLoader(dataset, batch_size, True, num_workers=0, collate_fn=pad_packed_collate)
        return dataloader
            
    def compute_stopword(self, y):
        res = torch.zeros_like(y).to(device=device)
        for i, row in enumerate(y):
            words_index = row.tolist()
            res[i] = torch.LongTensor([int(index in self.stop_words_index) for index in words_index])
        return res
    
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
    
    def __init__(self, src_seq_lens, src_seqs, trg_seqs, trg_stops, src_tfs, trg_ents, trg_ents_mask, trg_seqs_ori):
        self.src_seq_lens = src_seq_lens
        self.src_seqs = src_seqs
        self.trg_seqs = trg_seqs
        self.trg_stops = trg_stops
        self.src_tfs = src_tfs
        self.num_total_seqs = len(src_seqs)
        self.trg_ents = trg_ents
        self.trg_ents_mask = trg_ents_mask
        self.trg_seqs_ori = trg_seqs_ori
        
    def __getitem__(self, index):
        src_seq_len = self.src_seq_lens[index]
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        trg_stop = self.trg_stops[index]
        src_tf = self.src_tfs[index]
        trg_ent = self.trg_ents[index]
        trg_ent_mask = self.trg_ents_mask[index] 
        trg_seq_ori = self.trg_seqs_ori[index]
        return src_seq_len, src_seq, trg_seq, trg_stop, src_tf, trg_ent, trg_ent_mask, trg_seq_ori
    
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
    
    def hierarchical_merge(sequences, sequence_lengths):
        lengths = torch.stack(sequence_lengths)
        utterance_length = lengths.shape[1]
        padded_seqs = torch.zeros(len(sequences), utterance_length, lengths.max().item()).long()
        for i, seq in enumerate(sequences):
            word_end = max(lengths[i]).item()
            padded_seqs[i, :utterance_length, :word_end] = seq
        return padded_seqs, lengths
    
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    batch_data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences    
    src_seq_lens, src_seqs, trg_seqs, trg_stops, src_tfs, trg_ents, trg_ents_mask, trg_seqs_ori = zip(*batch_data)
    src_seqs, src_lens = hierarchical_merge(src_seqs, src_seq_lens)
    trg_seqs, trg_lens = merge(trg_seqs)
    trg_stops, _ = merge(trg_stops)

    trg_ents, _ = merge(trg_ents)
    trg_ents_mask, _ = merge(trg_ents)

    return (src_seqs.to(device=device), src_lens.to(device=device), 
            trg_seqs.to(device=device), trg_lens, 
            trg_stops.to(device=device), torch.stack(src_tfs).to(device=device), 
            trg_ents.to(device=device), trg_ents_mask.to(device=device), trg_seqs_ori)
    
