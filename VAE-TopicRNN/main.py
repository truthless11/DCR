# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:14:43 2018

@author: truthless
"""

from TopicRNN import TopicRNN
from TopicOptimizer import loss_function
from DataManager import DataManager
from Parser import getParser
from Metrics import blue, language_model_p, perplexity
import sys, os
from tqdm import tqdm
import torch
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

argv = sys.argv[1:]
parser = getParser()
args, _ = parser.parse_known_args(argv)
print(args)
with open(args.log + '.txt', 'a') as f:
    f.write('{0}\n'.format(str(args)))
torch.manual_seed(args.seed)

manager = DataManager(args.data)
train = manager.create_dataset('train', args.batch)
valid = manager.create_dataset('valid', args.batch)
test = manager.create_dataset('test', args.batch)

model = TopicRNN(args.rnn, len(manager.word2index), len(manager.index2nonstop), 
                 args.embed, args.rnn_dim, args.infer_dim, args.topic, args.teacher)
model.to(device=device)
if args.load != '':
    pretrain_model = torch.load(args.load) 
    model_dict = model.state_dict() 
    pretrained_dict = pretrain_model.state_dict() 
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict) 
optimizer = optim.Adam(model.parameters(), lr=args.lr)

if not args.test:
    if not os.path.exists(args.save):
        os.mkdir(args.save)
        
    for epoch in range(args.epoch):
        model.train()
        pbar = tqdm(enumerate(train), total=len(train))
        CE_loss, KLD_loss, SCE_loss = 0., 0., 0.
        for i, data in pbar:
            optimizer.zero_grad()
            outputs, word_p, indicator_p, mu, logvar = model(data[0], data[1], data[2], 
                                                             data[3], data[4], data[5])
            y_outputs = manager.compute_stopword(outputs)
            CE, KLD, SCE = loss_function(word_p, data[4], indicator_p, y_outputs, mu, logvar, data[5])
            loss = CE + KLD + SCE
            loss.backward()
            CE_loss += CE.item()
            KLD_loss += KLD.item()
            SCE_loss += SCE.item()
            optimizer.step()
            if (i+1) % args.print_per_batch == 0:
                CE_loss /= args.print_per_batch
                KLD_loss /= args.print_per_batch
                SCE_loss /= args.print_per_batch
                pbar.set_description('====> Iteration: {}, CE:{:.2f}, KLD:{:.2f}, SCE:{:.2f}'.format(i,
                                     CE_loss, KLD_loss, SCE_loss))
                CE_loss, KLD_loss, SCE_loss = 0., 0., 0.
        torch.save(model, "{0}/model_{1}_{2}".format(args.save, args.log, epoch))
        
        model.eval()
        pbar = tqdm(enumerate(valid), total=len(valid))
        CE_loss, KLD_loss, SCE_loss = 0., 0., 0.
        with torch.no_grad():
            for i, data in pbar:
                outputs, word_p, indicator_p, mu, logvar = model(data[0], data[1], data[2], 
                                                                 data[3], data[4], data[5], 
                                                                 training=False, use_teacher_forcing=False)
                y_outputs = manager.compute_stopword(outputs)
                CE, KLD, SCE = loss_function(word_p, data[4], indicator_p, y_outputs, mu, logvar, data[5])
                CE_loss += CE.item()
                KLD_loss += KLD.item()
                SCE_loss += SCE.item()
                if i == 0:
                    with open(args.log + '.txt', 'a') as f:
                        manager.interpret(outputs, data[4], data[5], f)
            CE_loss /= len(valid)
            KLD_loss /= len(valid)
            SCE_loss /= len(valid)
            pbar.set_description('====> Valid set loss, CE:{:.2f}, KLD:{:.2f}, SCE:{:.2f}'.format(CE_loss, 
                                 KLD_loss, SCE_loss))
        with open(args.log + '.txt', 'a') as f:
            f.write('Epoch:{:d}, CE:{:.2f}, KLD:{:.2f}, SCE:{:.2f}\n'.format(epoch, CE_loss, 
                                 KLD_loss, SCE_loss))
        
else:
    model.eval()
    pbar = tqdm(enumerate(test), total=len(test))
    blue_score, L, p, N = 0, 0, 0, 0
    with torch.no_grad():
        for i, data in pbar:
            outputs, word_p, indicator_p, mu, logvar = model(data[0], data[1], data[2], 
                                                             data[3], data[4], data[5],
                                                             training=False, use_teacher_forcing=False)
            blue_score += blue(data[4], outputs, data[5])
            L += len(data[5])
            outputs, word_p, indicator_p, mu, logvar = model(data[0], data[1], data[2], 
                                                             data[3], data[4], data[5],
                                                             training=False, use_teacher_forcing=True)
            p += language_model_p(data[4], word_p, data[5])
            N += sum(data[5])
    blue_score /= L
    perplexity_score = perplexity(p, N)
    print('Test set Blue:{:.4f}, Perplexity:{:.4f}'.format(blue_score, perplexity_score))
    