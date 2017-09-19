import torch
from torch.autograd import Variable


def get_batch(source, target, batch_size, seq_len=10, cuda=False, evalu=False):
    """Generate batch from the raw data."""
    nbatch = source.size(0) // batch_size
    # Trim extra elements doesn't fit well
    source = source.narrow(0, 0, nbatch*batch_size)
    target = target.narrow(0, 0, nbatch*batch_size)
    # Make batch shape
    source = source.view(batch_size, -1).t().contiguous()
    target = target.view(batch_size, -1).t().contiguous()
    if cuda:
        source, target = source.cuda(), target.cuda()
    for i in range(source.size(0) // seq_len):
        X = Variable(source[i*seq_len:(i+1)*seq_len], volatile=evalu)
        y = Variable(target[i*seq_len:(i+1)*seq_len])
        yield X, y

def repackage_hidden(h):
    """Wrap hidden in the new Variable to detach it from old history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
