import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import data.dict as dict
import math

class global_attention(nn.Module):

    def __init__(self, hidden_size, dropout=0.5, softmax=True):
        super(global_attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.linear_out = nn.Linear(2*hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        if softmax:
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.softmax = nn.Sigmoid()#(dim=-1)
        self.tanh = nn.Tanh()
        self.SELU = nn.Sequential(nn.SELU(), nn.AlphaDropout(p=0.05))

    def forward(self, x, context, conv=False):
        gamma_h = self.SELU(self.linear_in(x)).unsqueeze(2)    # batch * size * 1
        if conv:
            weights = torch.bmm(context, gamma_h).squeeze(2)
        else:
            weights = torch.bmm(context, gamma_h).squeeze(2)   # batch * time
        weights = self.softmax(weights)   # batch * time
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1) # batch * size
        output = self.SELU(self.linear_out(torch.cat([c_t, x], 1)))
        return output, weights


class self_attention(nn.Module):

    def __init__(self, hidden_size, dropout=0.1):
        super(self_attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.linear_in.weight)
        init.constant(self.linear_in.bias, 0.0)
        self.dropout1 = nn.Dropout(dropout)
        self.linear_out = nn.Linear(2*hidden_size, hidden_size)
        init.xavier_normal(self.linear_out.weight)
        init.constant(self.linear_in.bias, 0.0)
        self.dropout2 = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.SELU = nn.Sequential(nn.SELU(), nn.AlphaDropout(p=0.05))

    def forward(self, context):
        gamma_context = self.dropout1(self.linear_in(context.transpose(1,2)))    # N * L * C
        weights = torch.bmm(gamma_context, context)   # N * L * L
        weights = weights / math.sqrt(512)
        weights = self.softmax(weights)   # N * L * L
        c_t = torch.bmm(weights, context.transpose(1,2)) # N * L * C
        output = self.dropout2(self.linear_out(torch.cat([c_t, context.transpose(1,2)], -1)))
        output = self.SELU(output)#output * self.sigmoid(output)
        output = output.transpose(1,2)
        return output