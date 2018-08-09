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

import config


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
    #          13: {'label': 1, 'text': ['I', 'love', 'you', '.']}
    dat = list(map(lambda x: list(map(lambda y: dictionary.word2idx[y], x['text'])), data))
    # dat --> 12: {123(I), 243435(go), 243153(to), ...}
    #         13: {123(I), 9056(love), 8673(you), ...}

    # calculate the maximum length for learning.
    maxlen = 0
    for item in dat:
        maxlen = max(maxlen, len(item))
    maxlen = min(maxlen, 500)

    # If it exceeds, cut sentence. Fill padding if it remains.
    for i in range(len(data)):
        if maxlen < len(dat[i]):            # cut 'dat' to 500 words or less.
            dat[i] = dat[i][:maxlen]
        else:
            for j in range(maxlen - len(dat[i])):
                dat[i].append(dictionary.word2idx['<pad>']) # fill padding for remaining space.
        # print(data[i])

    targets = list(map(lambda x: x['label'], data))

    dat = Variable(torch.LongTensor(dat), volatile=volatile)
    targets = Variable(torch.LongTensor(targets), volatile=volatile)
    return dat.t(), targets

def print_prediction(data, output, attention):

    pred = []
    for i, each in enumerate(data):
        
        try:
            text_len = data[i].index(0)
        except ValueError:
            text_len = len(data[i])

        item = {
            'text': data[i][:text_len],
            'score': output[i],
            'atten': attention[i][0][:text_len]
        }
        pred.append(item)
    print(pred)

def evaluate():
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    for batch, i in enumerate(range(0, len(data_val), g_batch_size)):
        data, targets = package(data_val[i:min(len(data_val), i+g_batch_size)], volatile=True)
        if g_cuda:
            data = data.cuda()
            targets = targets.cuda()
        hidden = model.init_hidden(data.size(1))

        # print(data)
        output, attention = model.forward(data, hidden)

        print_prediction(data.t().cpu().detach().numpy().tolist(), 
            output.cpu().detach().numpy().tolist(), 
            attention.cpu().detach().numpy().tolist())

        # print('output: {}'.format(output))
        # print('attention: {}'.format(attention))
        print('data size: {}'.format(data.size()))
        print('output size: {}'.format(output.size()))
        print('attention size: {}'.format(attention.size()))
        output_flat = output.view(data.size(1), -1)
        total_loss += criterion(output_flat, targets).data
        prediction = torch.max(output_flat, 1)[1]
        total_correct += torch.sum((prediction == targets).float())
    return total_loss[0] / (len(data_val) // g_batch_size), total_correct.data[0] / len(data_val)


def train(epoch_number):
    
    global best_val_loss, best_acc
    model.train()       # Set the module in training mode.
    total_loss = 0
    total_pure_loss = 0  # without the penalization term
    start_time = time.time()

    for batch, i in enumerate(range(0, len(data_train), g_batch_size)):

        data, targets = package(data_train[i:i+g_batch_size], volatile=False)
        if g_cuda:
            data = data.cuda()
            targets = targets.cuda()

        hidden = model.init_hidden(data.size(1))
        output, attention = model.forward(data, hidden)
        loss = criterion(output.view(data.size(1), -1), targets)
        total_pure_loss += loss.data

        if attention is not None:  # add penalization term
            attentionT = torch.transpose(attention, 1, 2).contiguous()
            extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])    # bmm: batch matrix multiplication
            loss += g_penalization_coeff * extra_loss
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), g_clip)
        optimizer.step()

        total_loss += loss.data

        print('batch: {}, log_interval: {}, batch mod log_intererval: {}'.format(batch, g_log_interval, batch%g_log_interval))

        if batch % g_log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f} | pure loss {:5.4f}'.format(
                  epoch_number, batch, len(data_train) // g_batch_size,
                  elapsed * 1000 / g_log_interval, total_loss[0] / g_log_interval,
                  total_pure_loss[0] / g_log_interval))
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
        with open(g_save, 'wb') as f:
            torch.save(model, f)
        f.close()
        best_val_loss = val_loss
    else:  # if loss doesn't go down, divide the learning rate by 5.
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2
    if not best_acc or acc > best_acc:
        with open(g_save[:-3]+'.best_acc.pt', 'wb') as f:
            torch.save(model, f)
        f.close()
        best_acc = acc
    with open(g_save[:-3]+'.epoch-{:02d}.pt'.format(epoch_number), 'wb') as f:
        torch.save(model, f)
    f.close()

def report_test_result(message):
    print('-' * 89)
    print(message)
    data_val = open(g_test_data).readlines()
    evaluate_start_time = time.time()
    test_loss, acc = evaluate()
    print('-' * 89)
    fmt = '| test | time: {:5.2f}s | test loss (pure) {:5.4f} | Acc {:8.4f}'
    print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))
    print('-' * 89)

def set_data(args):

    global g_emsize, g_nhid, g_nlayers, g_attention_unit, g_attention_hops, \
            g_dropout, g_clip, g_nfc, g_lr, g_epochs, g_seed, g_cuda, g_log_interval, \
            g_save, g_dictionary, g_word_vector, g_train_data, g_val_data, \
            g_test_data, g_batch_size, g_class_number, g_optimizer, g_penalization_coeff

    g_emsize = args['emsize']
    g_nhid = args['nhid']
    g_nlayers = args['nlayers']
    g_attention_unit = args['attention-unit']
    g_attention_hops = args['attention-hops']
    g_dropout = args['dropout']
    g_clip = args['clip']
    g_nfc = args['nfc']
    g_lr = args['lr']
    g_epochs = args['epochs']
    g_seed = args['seed']
    g_cuda = args['cuda']
    g_log_interval = args['log-interval']
    g_save = args['save']
    g_dictionary = args['dictionary']
    g_word_vector = args['word-vector']
    g_train_data = args['train-data']
    g_val_data = args['val-data']
    g_test_data = args['test-data']
    g_batch_size = args['batch-size']
    g_class_number = args['class-number']
    g_optimizer = args['optimizer']
    g_penalization_coeff = args['penalization-coeff']

if __name__ == '__main__':

    # parse the arguments
    # args = get_args()
    args = config.arg_sst
    set_data(args)
    
    # Set the random seed manually for reproducibility.
    torch.manual_seed(g_seed)
    if torch.cuda.is_available():
        if not g_cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(g_seed)
    random.seed(g_seed)

    # Load Dictionary
    assert os.path.exists(g_train_data)
    assert os.path.exists(g_val_data)
    print('Begin to load the dictionary.')
    dictionary = Dictionary(path=g_dictionary)

    best_val_loss = None
    best_acc = None

    n_token = len(dictionary)
    model = Classifier({
        'dropout': g_dropout,
        'ntoken': n_token,                    # the number of words
        'nlayers': g_nlayers,                 # the number of hidden layers in Bi-LSTM. default 2.
        'nhid': g_nhid,                       # hidden layer size per layer
        'ninp': g_emsize,                     # word embedding size default 300
        'pooling': 'all',                     # 'all': use the Self-Attentive Encoder
        'attention-unit': g_attention_unit,   # attention unit number, d_a in the paper, default 350
        'attention-hops': g_attention_hops,   # hop number, r in the paper, default 1
        'nfc': g_nfc,                         # hidden layer size for MLP in the classifier, default 512
        'dictionary': dictionary,             # location of the dictionary generated by the tokenizer
        'word-vector': g_word_vector,         # location of the initial word vector, e.g. GloVe, should be a torch .pt model
        'class-number': g_class_number        # number of class for the last step of classification
    })
    
    if g_cuda:
        model = model.cuda()
    
    # Identity matrix for frobenious norm regularization
    I = Variable(torch.zeros(g_batch_size, g_attention_hops, g_attention_hops))
    for i in range(g_batch_size):
        for j in range(g_attention_hops):
            I.data[i][j][j] = 1
    if g_cuda:
        I = I.cuda()

    # Loss function: CrossEntropyLoss, Optimizer: default Adam
    criterion = nn.CrossEntropyLoss()
    if g_optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=g_lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    elif g_optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=g_lr, momentum=0.9, weight_decay=0.01)
    else:
        raise Exception('For other optimizers, please add it yourself. '
                        'supported ones are: SGD and Adam.')
    
    # Load data
    print('Begin to load data.')
    data_train = open(g_train_data).readlines()
    data_val = open(g_val_data).readlines()
    
    try:
        for epoch in range(g_epochs):
            train(epoch)

        report_test_result('Final test result.')
    except KeyboardInterrupt:
        report_test_result('Exit from training early.')
    
    exit(0)