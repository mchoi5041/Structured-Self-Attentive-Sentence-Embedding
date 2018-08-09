from __future__ import print_function
from models import *

from util import Dictionary, get_args

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import json
import time
import random
import os


def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 2), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


def package(data, volatile=False):

    """Package data for training / evaluation."""
    data = list(map(lambda x: json.loads(x), data))     
    # data --> 12: {'label': 1, 'text': ['I', 'go', 'to', 'the', 'school', '.']}
    dat = list(map(lambda x: list(map(lambda y: dictionary.word2idx[y], x['text'])), data))
    # dat --> 12: {123(I), 243435(go), 243153(to), ...}

    maxlen = 0
    for item in dat:
        maxlen = max(maxlen, len(item))
    targets = list(map(lambda x: x['label'], data))
    maxlen = min(maxlen, 500)
    for i in range(len(data)):
        if maxlen < len(dat[i]):            # cut 'dat' to 500 words or less.
            dat[i] = dat[i][:maxlen]
        else:
            for j in range(maxlen - len(dat[i])):
                dat[i].append(dictionary.word2idx['<pad>']) # fill padding for remaining space.

    dat = Variable(torch.LongTensor(dat), volatile=volatile)
    targets = Variable(torch.LongTensor(targets), volatile=volatile)
    return dat.t(), targets


def evaluate():
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    for batch, i in enumerate(range(0, len(data_val), args.batch_size)):
        data, targets = package(data_val[i:min(len(data_val), i+args.batch_size)], volatile=True)
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()
        hidden = model.init_hidden(data.size(1))
        output, attention = model.forward(data, hidden)
        output_flat = output.view(data.size(1), -1)
        total_loss += criterion(output_flat, targets).data
        prediction = torch.max(output_flat, 1)[1]
        total_correct += torch.sum((prediction == targets).float())
    return total_loss[0] / (len(data_val) // args.batch_size), total_correct.data[0] / len(data_val)


def train(epoch_number):
    
    global best_val_loss, best_acc
    model.train()       # Set the module in training mode.
    total_loss = 0
    total_pure_loss = 0  # without the penalization term
    start_time = time.time()

    for batch, i in enumerate(range(0, len(data_train), args.batch_size)):

        data, targets = package(data_train[i:i+args.batch_size], volatile=False)
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()

        hidden = model.init_hidden(data.size(1))
        output, attention = model.forward(data, hidden)
        loss = criterion(output.view(data.size(1), -1), targets)
        total_pure_loss += loss.data

        if attention is not None:  # add penalization term
            attentionT = torch.transpose(attention, 1, 2).contiguous()
            extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])    # bmm: batch matrix multiplication
            loss += args.penalization_coeff * extra_loss
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data

        print('batch: {}, log_interval: {}, batch mod log_intererval: {}'.format(batch, args.log_interval, batch%args.log_interval))

        if batch % args.log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f} | pure loss {:5.4f}'.format(
                  epoch_number, batch, len(data_train) // args.batch_size,
                  elapsed * 1000 / args.log_interval, total_loss[0] / args.log_interval,
                  total_pure_loss[0] / args.log_interval))
            total_loss = 0
            total_pure_loss = 0
            start_time = time.time()

#            for item in model.parameters():
#                print item.size(), torch.sum(item.data ** 2), torch.sum(item.grad ** 2).data[0]
#            print model.encoder.ws2.weight.grad.data
#            exit()
    
    evaluate_start_time = time.time()
    val_loss, acc = evaluate()
    print('-' * 89)
    fmt = '| evaluation(epoch: {}) | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f}'
    print(fmt.format( epoch_number, (time.time() - evaluate_start_time), val_loss, acc))
    print('-' * 89)

    # Save the model, if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        f.close()
        best_val_loss = val_loss
    else:  # if loss doesn't go down, divide the learning rate by 5.
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2
    if not best_acc or acc > best_acc:
        with open(args.save[:-3]+'.best_acc.pt', 'wb') as f:
            torch.save(model, f)
        f.close()
        best_acc = acc
    with open(args.save[:-3]+'.epoch-{:02d}.pt'.format(epoch_number), 'wb') as f:
        torch.save(model, f)
    f.close()

def report_test_result(message):
    print('-' * 89)
    print(message)
    data_val = open(args.test_data).readlines()
    evaluate_start_time = time.time()
    test_loss, acc = evaluate()
    print('-' * 89)
    fmt = '| test | time: {:5.2f}s | test loss (pure) {:5.4f} | Acc {:8.4f}'
    print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))
    print('-' * 89)

if __name__ == '__main__':

    # parse the arguments
    args = get_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # Load Dictionary
    assert os.path.exists(args.train_data)
    assert os.path.exists(args.val_data)
    print('Begin to load the dictionary.')
    dictionary = Dictionary(path=args.dictionary)

    best_val_loss = None
    best_acc = None

    n_token = len(dictionary)
    model = Classifier({
        'dropout': args.dropout,
        'ntoken': n_token,          # the number of words
        'nlayers': args.nlayers,    # the number of hidden layers in Bi-LSTM. default 2.
        'nhid': args.nhid,          # hidden layer size per layer
        'ninp': args.emsize,        # word embedding size default 300
        'pooling': 'all',           # 'all': use the Self-Attentive Encoder
        'attention-unit': args.attention_unit,  # attention unit number, d_a in the paper, default 350
        'attention-hops': args.attention_hops,  # hop number, r in the paper, default 1
        'nfc': args.nfc,            # hidden layer size for MLP in the classifier, default 512
        'dictionary': dictionary,   # location of the dictionary generated by the tokenizer
        'word-vector': args.word_vector,    # location of the initial word vector, e.g. GloVe, should be a torch .pt model
        'class-number': args.class_number   # number of class for the last step of classification
    })
    print(args)
    if args.cuda:
        model = model.cuda()
    
    # Identity matrix for frobenious norm regularization
    I = Variable(torch.zeros(args.batch_size, args.attention_hops, args.attention_hops))
    for i in range(args.batch_size):
        for j in range(args.attention_hops):
            I.data[i][j][j] = 1
    if args.cuda:
        I = I.cuda()

    # Loss function: CrossEntropyLoss, Optimizer: default Adam
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
    else:
        raise Exception('For other optimizers, please add it yourself. '
                        'supported ones are: SGD and Adam.')
    
    # Load data
    print('Begin to load data.')
    data_train = open(args.train_data).readlines()
    data_val = open(args.val_data).readlines()
    
    try:
        for epoch in range(args.epochs):
            train(epoch)

        report_test_result('Final test result.')
    except KeyboardInterrupt:
        report_test_result('Exit from training early.')
    
    exit(0)