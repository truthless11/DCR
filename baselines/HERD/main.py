# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:14:43 2018

@author: truthless
"""

from HERD import HERD
from Optimizer import loss_function
from DataManager import DataManager
from Parser import getParser
from Metrics import bleu, language_model_p, perplexity, entity_recall
import sys, os
from tqdm import tqdm
import torch
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

argv = sys.argv[1:]
parser = getParser()
args, _ = parser.parse_known_args(argv)
print(args)
with open('{0}.txt'.format(args.log), 'a') as f:
    f.write('{0}\n'.format(str(args)))
torch.manual_seed(args.seed)

manager = DataManager(args.data, args.no_wordvec, args.embed, args.context_len)
if not args.test:
    train = manager.create_dataset('train', args.batch)
    valid = manager.create_dataset('valid', args.batch)
else:
    test = manager.create_dataset('test', args.batch)

model = HERD(args.rnn, manager.vector, len(manager.word2index), args.embed, args.rnn_dim, args.teacher)
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
        CE_loss = 0.
        for i, data in pbar:
            optimizer.zero_grad()
            outputs, word_p = model(data[0], data[1], data[2], data[3])
            CE = loss_function(word_p, data[2], data[3])
            loss = CE
            loss.backward()
            CE_loss += CE.item()
            optimizer.step()
            if (i+1) % args.print_per_batch == 0:
                CE_loss /= args.print_per_batch
                pbar.set_description('====> Iteration: {}, CE:{:.2f}'.format(i, CE_loss))
                CE_loss = 0.
        if (epoch+1) % args.save_per_batch == 0:
            torch.save(model, "{0}/model_{1}_{2}".format(args.save, args.log, epoch))
        
        model.eval()
        pbar = tqdm(enumerate(valid), total=len(valid))
        CE_loss = 0.
        bleu_score, L, p, N = 0, 0, 0, 0
        with torch.no_grad():
            for i, data in pbar:
                outputs, word_p = model(data[0], data[1], data[2], data[3], use_teacher_forcing=False)
                CE = loss_function(word_p, data[2], data[3])
                CE_loss += CE.item()
                if i == 0:
                    with open('{0}.txt'.format(args.log), 'a') as f:
                        manager.interpret(outputs, data[2], data[3], f)
                bleu_score += bleu(data[2], outputs, data[3])
                L += len(data[3])
                outputs, word_p = model(data[0], data[1], data[2], data[3], use_teacher_forcing=True)
                p += language_model_p(data[2], word_p, data[3])
                N += sum(data[3])
            CE_loss /= len(valid)
            print('Valid set loss, CE:{:.2f}'.format(CE_loss))
            bleu_score /= L
            perplexity_score = perplexity(p, N)
            print('Valid set metrics, Bleu:{:.4f}, Perplexity:{:.4f}'.format(bleu_score, perplexity_score))
        with open('{0}.txt'.format(args.log), 'a') as f:
            f.write('Epoch:{:d}, CE:{:.2f}\n'.format(epoch, CE_loss))
            f.write('Bleu:{:.4f}, Perplexity:{:.4f}\n'.format(bleu_score, perplexity_score))
        
else:
    model.eval()
    pbar = tqdm(enumerate(test), total=len(test))
    bleu_score, L, p, N = 0, 0, 0, 0
    acc, cnt = 0, 0
    with torch.no_grad():
        for i, data in pbar:
            outputs, word_p = model(data[0], data[1], data[2], data[3], use_teacher_forcing=False)
            bleu_score += bleu(data[2], outputs, data[3])
            L += len(data[3])
            outputs, word_p = model(data[0], data[1], data[2], data[3], use_teacher_forcing=True)
            p += language_model_p(data[2], word_p, data[3])
            N += sum(data[3])
            acc_batch, cnt_batch = entity_recall(data[4], outputs)
            acc += acc_batch
            cnt += cnt_batch
    bleu_score /= L
    perplexity_score = perplexity(p, N)
    ent_recall = acc / cnt
    print('Test set metrics, Bleu:{:.4f}, Perplexity:{:.4f}, Entity Recall:{:.4f}'.format(
            bleu_score, perplexity_score, ent_recall))
    