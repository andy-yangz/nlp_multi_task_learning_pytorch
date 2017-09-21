import torch.nn as nn
from torch.autograd import Variable


class EncoderModel(nn.Module):
    """Model include a transducer to predict at each time steps"""

    def __init__(self, ntoken, emsize, nhid,
                 nlayers=1, dropout=0.2, rnn_type='LSTM', bi=False):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(ntoken, emsize)
        self.rnn_type = rnn_type

        # Select RNN cell type from LSTM, GRU, and Elman
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(emsize, nhid, nlayers, bidirectional=bi)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(emsize, nhid, nlayers, bidirectional=bi)
        else:
            self.rnn = nn.RNN(emsize, nhid, nlayers, bidirectional=bi)
    
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers
        self.bi = bi
        
    def init_weights(self):
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, hidden):
        embeded = self.drop(self.embed(input))
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(embeded, hidden)
        output = self.drop(output)
        return output, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers*(1+int(self.bi)), 
                                    batch_size, self.nhid).zero_()),
                Variable(weight.new(self.nlayers*(1+int(self.bi)), 
                                    batch_size, self.nhid).zero_()))
                    

class LinearDecoder(nn.Module):
    """Linear decoder to decoder the outputs from the RNN Encoder.
        Then we can get the results of different tasks."""

    def __init__(self, nhid, ntags, bi=False):
        super().__init__()
        self.linear = nn.Linear(nhid*(1+int(bi)), ntags)
        self.init_weights()
        self.nin = nhid
        self.nout = ntags
        self.bi = bi
    
    def init_weights(self):
        init_range = 0.1
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, input):
        logit = self.linear(input.view(input.size(0)*input.size(1), input.size(2)))
        return logit.view(input.size(0), input.size(1), logit.size(1))


class JointModel(nn.Module):
    """Joint Model to joint training two tasks.
       You can also only select one train mode to train one task.
       For args to specified the detail of training, include the task
       output and which layer we put it in. Number of tag first and 
       then number of layer."""
    def __init__(self, ntoken, emsize, nhid, *args,
                 dropout=0.2, rnn_type='LSTM', bi=False, train_mode='Joint'):
        super().__init__()
        self.ntoken = ntoken
        self.emsize = emsize
        self.nhid = nhid
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.bi = bi
        self.train_mode = train_mode
        # According to train type, take arguments
        if train_mode == 'Joint':
            self.ntags1 = args[0]
            self.nlayers1 = args[1]
            self.ntags2 = args[2]
            self.nlayers2 = args[3]
            if self.nlayers1 == self.nlayers2:
                self.rnn = EncoderModel(ntoken, emsize, nhid, self.nlayers1, 
                                        dropout, rnn_type, bi)
            else:
                # Lower layer
                self.rnn1 = EncoderModel(ntoken, emsize, nhid, self.nlayers1, 
                                         dropout, rnn_type, bi)
                # Higher layer
                if rnn_type == 'LSTM':
                    self.rnn2 = nn.LSTM(nhid*(1+int(bi)), nhid, 
                                        self.nlayers2 - self.nlayers1, 
                                        bidirectional=bi)
                elif rnn_type == 'GRU':
                    self.rnn2 = nn.GRU(nhid*(1+int(bi)), nhid, 
                                       self.nlayers2 - self.nlayers1, 
                                       bidirectional=bi)
                else:
                    self.rnn2 = nn.RNN(nhid*(1+int(bi)), nhid, 
                                       self.nlayers2 - self.nlayers1, 
                                       bidirectional=bi)

            # Decoders for two tasks
            self.linear1 = LinearDecoder(nhid, self.ntags1, bi)
            self.linear2 = LinearDecoder(nhid, self.ntags2, bi)
            
        else:
            self.ntags = args[0]
            self.nlayers = args[1]
            self.rnn = EncoderModel(ntoken, emsize, nhid, self.nlayers, 
                                    dropout, rnn_type, bi)
            self.linear = LinearDecoder(nhid, self.ntags, bi)
        
    def forward(self, input, *hidden):
        if self.train_mode == 'Joint':
            if self.nlayers1 == self.nlayers2:
                logits, hidden = self.rnn(input, hidden[0])
                outputs1 = self.linear1(logits)
                outputs2 = self.linear2(logits)
                return outputs1, outputs2, hidden
            else:
                logits1, hidden1 = self.rnn1(input, hidden[0])
                self.rnn2.flatten_parameters()
                logits2, hidden2 = self.rnn2(logits1, hidden[1])
                outputs1 = self.linear1(logits1)
                outputs2 = self.linear2(logits2)
                return outputs1, outputs2, hidden1, hidden2
        else:
            logits, hidden = self.rnn(input, hidden[0])            
            outputs = self.linear(logits)
            return outputs, hidden

    def init_rnn2_hidden(self, batch_size):
        weight = next(self.rnn2.parameters()).data
        return (Variable(weight.new((self.nlayers2 - self.nlayers1)*(1+int(self.bi)), 
                                    batch_size, self.nhid).zero_()),
                Variable(weight.new((self.nlayers2 - self.nlayers1)*(1+int(self.bi)), 
                                    batch_size, self.nhid).zero_()))
        