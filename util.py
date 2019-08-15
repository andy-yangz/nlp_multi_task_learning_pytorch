import torch
from torch.autograd import Variable


def get_batch(source, *targets, batch_size, seq_len=10, cuda=False, evalu=False):
    """Generate batch from the raw data."""
    nbatch = source.size(0) // batch_size
    shuffle_mask = torch.randperm(batch_size)
    # Trim extra elements doesn't fit well
    source = source.narrow(0, 0, nbatch*batch_size)
    # Make batch shape
    source = source.view(batch_size, -1).t().contiguous()
    # Shuffle 
    source = source[:, shuffle_mask]
    if cuda:
        source = source.cuda()
    
    targets = list(targets)
    for i in range(len(targets)):
        targets[i] = targets[i].narrow(0, 0, nbatch*batch_size)
        targets[i] = targets[i].view(batch_size, -1).t().contiguous()
        targets[i] = targets[i][:, shuffle_mask]
        if cuda:
            targets[i] = targets[i].cuda()
    
    for i in range(source.size(0) // seq_len):
        ys = []

        if evalu is True:
            #print ("evalu is True")
            with torch.no_grad():
                #print ("torch.no_grad for X variable")
                X = Variable(source[i*seq_len:(i+1)*seq_len])
        else:
            #print ("evalu is False. x should be trainable")
            X = Variable(source[i*seq_len:(i+1)*seq_len])

        for target in targets:
            ys.append(Variable(target[i*seq_len:(i+1)*seq_len]))
        yield X, ys

def repackage_hidden(h):
    """Wrap hidden in the new Variable to detach it from old history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
