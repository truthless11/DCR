# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:14:43 2018

@author: truthless
"""

from TopicRNN import TopicRNN
from TopicOptimizer import loss_function
from DataManager import DataManager, create_dataset
from Parser import getParser
import sys
from tqdm import tqdm
import torch
from torch import optim

argv = sys.argv[1:]
parser = getParser()
args, _ = parser.parse_known_args(argv)
torch.manual_seed(args.seed)

manager = DataManager(args.datapath)
train = create_dataset(manager.data['train'], manager.stop_words_index, args.batch)
valid = create_dataset(manager.data['valid'], manager.stop_words_index, args.batch)
test = create_dataset(manager.data['test'], manager.stop_words_index, args.batch)

model = TopicRNN(args.rnn, len(manager.word2index)+4, args.embed, args.rnn_dim, args.infer_dim, args.topic, args.teacher)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

if not args.test:
    for epoch in range(args.epoch):
        model.train()
        pbar = tqdm(enumerate(train), total=len(train))
        CE_loss, KLD_loss, SCE_loss = 0., 0., 0.
        for i, data in pbar:
            outputs, word_p, indicator_p, mu, logvar = model(data[0], data[1], data[2], data[3], data[4])
            y_outputs = manager.compute_stopword(outputs)
            CE, KLD, SCE = loss_function(word_p, data[3], indicator_p, y_outputs, mu, logvar, data[4])
            loss = CE + KLD + SCE
            loss.backward()
            CE_loss += CE.item()
            KLD_loss += KLD.item()
            SCE_loss += SCE.item()
            if (i+1) % args.print_per_batch == 0:
                CE_loss /= args.print_per_batch
                KLD_loss /= args.print_per_batch
                SCE_loss /= args.print_per_batch
                pbar.set_description('====> Epoch: {}, CE:{:.2f}, KLD:{:.2f}, SCE:{:.2f}'.format(i, CE_loss, KLD_loss, SCE_loss))
                CE_loss, KLD_loss, SCE_loss = 0., 0., 0.
        
        model.eval()
        pbar = tqdm(enumerate(valid), total=len(valid))
        CE_loss, KLD_loss, SCE_loss = 0., 0., 0.
        with torch.no_grad():
            for i, data in pbar:
                outputs, word_p, indicator_p, mu, logvar = model(data[0], data[1], data[2], data[3], data[4])
                y_outputs = manager.compute_stopword(outputs)
                CE, KLD, SCE = loss_function(word_p, data[3], indicator_p, y_outputs, mu, logvar, data[4])
                CE_loss += CE.item()
                KLD_loss += KLD.item()
                SCE_loss += SCE.item()
            CE_loss /= len(valid)
            KLD_loss /= len(valid)
            SCE_loss /= len(valid)
            pbar.set_description('====> Valid set loss, CE:{:.2f}, KLD:{:.2f}, SCE:{:.2f}'.format(CE_loss, KLD_loss, SCE_loss))
else:
    pabr = tqdm(enumerate(test), total=len(test))
    #TODO
    
    