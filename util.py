import torch
from random import shuffle
from torch.autograd import Variable

PAD_token = 0


def get_batch(source, pos_target, dep_target, batch_size, cuda=False):
    """A generator to get batch for the data."""
    tmp = list(zip(source, pos_target, dep_target))
    shuffle(tmp)
    source, pos_target, dep_target = zip(*tmp)

    n_batch = (len(source) // batch_size) + 1
    for i in range(n_batch):
        X = source[i*batch_size : (i+1)*batch_size]
        X = Variable(torch.LongTensor(pad_batch(X)))
        y_pos = pos_target[i*batch_size : (i+1)*batch_size]
        y_pos = Variable(torch.LongTensor(pad_batch(y_pos)))
        y_dep = dep_target[i*batch_size : (i+1)*batch_size]
        y_dep = Variable(torch.LongTensor(pad_dep_batch(y_dep)))
        if cuda:
            X = X.cuda()
            y_pos = y_pos.cuda()
            y_dep = y_dep.cuda()
        yield X, y_pos, y_dep


def pad_dep_batch(dep_batch):
    max_length = max([len(seq) for seq in dep_batch])
    pad_dep_batch = [pad_dep_seq(seq, max_length) for seq in dep_batch]
    return pad_dep_batch

def pad_dep_seq(seq, max_length):
    seq += [[0, PAD_token] for i in range(max_length - len(seq))]
    return seq   

def pad_batch(batch):
    max_length = max([len(seq) for seq in batch])
    pad_batch = [pad_seq(seq, max_length) for seq in batch]
    return pad_batch

def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def repackage_hidden(h):
    """Wrap hidden in the new Variable to detach it from old history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
