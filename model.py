import torch.nn as nn
from torch.autograd import Variable


class RNNModel(nn.Module):
    """Model include a transducer to predict at each time steps"""

    def __init__(self, ntoken, nout, ninp, nhid,
                 nlayers=1, dropout=0.5, rnn_type='LSTM',bi=False):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers)
        self.linear = nn.Linear(nhid*(1+int(bi)), nout)

        self.nhid = nhid
        self.nlayers = nlayers
        self.bi = bi
        
    def init_weights(self):
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.fill_(0)
        self.linera.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, hidden):
        embeded = self.drop(self.embed(input))
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(embeded, hidden)
        output = self.drop(output)
        logit = self.linear(output.view(output.size(0)*output.size(1), output.size(2)))
        return logit.view(output.size(0), output.size(1), logit.size(1)), hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers*(1+int(self.bi)), 
                                    batch_size, self.nhid).zero_()),
                Variable(weight.new(self.nlayers*(1+int(self.bi)), 
                                    batch_size, self.nhid).zero_()))