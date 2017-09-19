import argparse
import os
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from data import Corpus
from util import *
from model import *

###############################################################################
# Set Parameters
###############################################################################
parser = argparse.ArgumentParser(description='Pytorch NLP multi-task leraning for POS tagging and chunking.')
parser.add_argument('--data', type=str, default='./data',
                    help='data file')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings') 
parser.add_argument('--nlayers', type=int, default=2, 
                    help='number of layers')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clip')
parser.add_argument('--epochs', type=int, default=40,
                    help='epoch number')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=15,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout rate')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--bi', action='store_true',
                    help='use bidirection RNN')
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()


###############################################################################
# Load Data
###############################################################################
corpus_path = args.save.strip() + '_corpus.pt'
print('Loading corpus...')
if os.path.exists(corpus_path):
    corpus = torch.load(corpus_path)
else:
    corpus = Corpus(args.data)
    torch.save(corpus, corpus_path)
###############################################################################
# Build Model
###############################################################################
nwords = corpus.word_dict.nwords
ntags = corpus.pos_dict.nwords
model = RNNModel(nwords, ntags, args.emsize, args.nhid, 
                 args.nlayers, args.dropout, bi=args.bi)
if args.cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

###############################################################################
# Training
###############################################################################

def train(loss_log):
    # Turn on training mode
    total_loss = 0
    start_time = time.time()
    n_iteration = corpus.word_train.size(0) // (args.batch_size*args.seq_len) 
    iteration = 0
    for X, y in get_batch(corpus.word_train, corpus.pos_train,
                          args.batch_size, args.seq_len, args.cuda):
        iteration += 1
        model.zero_grad()

        hidden = model.init_hidden(args.batch_size)
        outputs, hidden = model(X, hidden)
        loss = criterion(outputs.view(-1, ntags), y.view(-1))
        loss.backward()

        # Prevent the exploding gradient
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data
        
        if iteration % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            cur_loss = cur_loss.cpu().numpy()[0]
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} iteration | {:5.2f} ms/batch | loss {:5.2f} |'.format(
                  epoch, iteration, n_iteration,
                  elapsed*1000/args.log_interval, 
                  cur_loss))
            loss_log.append(cur_loss)
            total_loss = 0
            start_time = time.time()
    return loss_log

def evaluate(source, target):
    model.eval()
    n_iteration = source.size(0) // (args.batch_size*args.seq_len)
    total_loss = 0
    hidden = model.init_hidden(args.batch_size)
    for X_val, y_val in get_batch(source, target, args.batch_size,
                                  args.seq_len, args.cuda, evalu=True):
        outputs, hidden = model(X_val, hidden)
        total_loss += criterion(outputs.view(-1, ntags), y_val.view(-1))
        _, pred = outputs.data.topk(1)
        # print(pred.size(), y_val.size())
        accuracy = torch.sum(pred.squeeze(2) == y_val.data) / (y_val.size(0) * y_val.size(1)) 
        hidden = repackage_hidden(hidden)
    return total_loss/n_iteration, accuracy


# Loop over epochs
best_val_loss = None
best_accuracy = None
best_epoch = 0
loss_log =[]
early_stop_count = 0
# You can break training early by Ctr+C
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        print('Begin training...')
        loss_log = train(loss_log)
        val_loss, accuracy = evaluate(corpus.word_valid, corpus.pos_valid)
        print('-'*50)
        print('| end of epoch {:3d} | valid loss {:5.2f} | accuracy {:5.2f} |'.format(
            epoch, val_loss.data.cpu().numpy()[0], accuracy
        ))
        if not best_val_loss or (val_loss.data[0] < best_val_loss):
            with open(args.save.strip() + '.pt', 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss.data[0]
        if not best_accuracy or (accuracy > best_accuracy):
            best_accuracy = accuracy
            best_epoch = epoch
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count >= 20:
            print('\nEarly Stopping! \nBecause 20 epochs the accuracy have no improvement.')
            break
except KeyboardInterrupt:
    print('-'*50)
    print('Exiting from training early.')


#Load the best saved model
with open(args.save.strip() + '.pt', 'rb') as f:
    model = torch.load(f)

test_loss, test_accuracy = evaluate(corpus.word_test, corpus.pos_test)
print('='*50)
print('| End of training | test loss: {:5.2f} | test acccuracy: {:5.2f}'.format(
    test_loss.data.cpu().numpy()[0], test_accuracy
))

# Save results
results = {
    'corpus': corpus,
    'best_epoch': best_epoch,
    'best_val_loss': best_val_loss,
    'best_accuracy': best_accuracy,
    'test_accuracy': test_accuracy,
    'loss_log': loss_log
}
torch.save(results, '%s_emsize%d_nlayers%d_nhid%d_dropout%3.1f_seqlen%d_bi%d_result.pt' \
                    %(args.save.strip(), args.emsize, args.nlayers, args.nhid,
                      args.dropout, args.seq_len, args.bi))

            
