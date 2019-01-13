# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:14:43 2018

@author: truthless
"""

from TopicRNN import TopicRNN
from TopicOptimizer import loss_function
from DataManager import DataManager
from Parser import getParser
from Metrics import bleu, language_model_p, perplexity
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

manager = DataManager(args.data, args.no_wordvec, args.embed)
if not args.test:
	train = manager.create_dataset('train', args.batch)
	valid = manager.create_dataset('valid', args.batch)
else:
	test = manager.create_dataset('test', args.batch)

model = TopicRNN(args.rnn, manager.vector, len(manager.word2index), len(manager.index2nonstop), 
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
                                                             data[3], data[5])
            CE, KLD, SCE = loss_function(word_p, data[2], indicator_p, data[4], mu, logvar, data[3])
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
        if (epoch+1) % 10 == 0:
            torch.save(model, "{0}/model_{1}_{2}".format(args.save, args.log, epoch))
        
        model.eval()
        pbar = tqdm(enumerate(valid), total=len(valid))
        CE_loss, KLD_loss, SCE_loss = 0., 0., 0.
        with torch.no_grad():
            for i, data in pbar:
                outputs, word_p, indicator_p, mu, logvar = model(data[0], data[1], data[2],
                                                                 data[3], data[5],
                                                                 use_teacher_forcing=False)
                CE, KLD, SCE = loss_function(word_p, data[2], indicator_p, data[4], mu, logvar, data[3])
                CE_loss += CE.item()
                KLD_loss += KLD.item()
                SCE_loss += SCE.item()
                if i == 0:
                    with open('{0}.txt'.format(args.log), 'a') as f:
                        manager.interpret(outputs, data[2], data[3], f)
            CE_loss /= len(valid)
            KLD_loss /= len(valid)
            SCE_loss /= len(valid)
            pbar.set_description('====> Valid set loss, CE:{:.2f}, KLD:{:.2f}, SCE:{:.2f}'.format(CE_loss, 
                                 KLD_loss, SCE_loss))
        with open('{0}.txt'.format(args.log), 'a') as f:
            f.write('Epoch:{:d}, CE:{:.2f}, KLD:{:.2f}, SCE:{:.2f}\n'.format(epoch, CE_loss, 
                                 KLD_loss, SCE_loss))
        
else:
    model.eval()
    pbar = tqdm(enumerate(test), total=len(test))
    bleu_score, L, p, N = 0, 0, 0, 0
    with torch.no_grad():
        for i, data in pbar:
            outputs, word_p, indicator_p, mu, logvar = model(data[0], data[1], data[2],
                                                             data[3], data[5],
                                                             use_teacher_forcing=False)
            bleu_score += bleu(data[2], outputs, data[3])
            L += len(data[3])
            outputs, word_p, indicator_p, mu, logvar = model(data[0], data[1], data[2],
                                                             data[3], data[5],
                                                             use_teacher_forcing=True)
            p += language_model_p(data[2], word_p, data[3])
            N += sum(data[3])
    bleu_score /= L
    perplexity_score = perplexity(p, N)
    print('Test set Blue:{:.4f}, Perplexity:{:.4f}'.format(bleu_score, perplexity_score))
	
    #draw topic words
    with torch.no_grad():
        for i in range(args.topic):
            theta = torch.zeros(1, args.topic).to(device=device)
            theta[0, i] = 1
            topic_additions = model.topic_decoder(theta)
            topv, topi = topic_additions[0].topk(20)
            with open('{0}_topic.txt'.format(args.log), 'a') as f:
                for index in topi:
                    f.write('{0} '.format(manager.index2word[index.item()]))
                f.write('\n')
    