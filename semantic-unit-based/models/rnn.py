import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import data.dict as dict
import models
import math


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, config, max_len=5000):
        pe = torch.zeros(max_len, config.emb_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.emb_size, 2) *
                             -(math.log(10000.0) / config.emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=config.dropout)
        self.emb_size = config.emb_size

    def forward(self, emb):
        # We must wrap the self.pe in Variable to compute, not the other
        # way - unwrap emb(i.e. emb.data). Otherwise the computation
        # wouldn't be watched to build the compute graph.
        emb = emb * math.sqrt(self.emb_size)
        # print(self.pe.size())
        emb = emb + Variable(self.pe[:emb.size(0)], requires_grad=False)
        emb = self.dropout(emb)
        return emb


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            lstm = nn.LSTMCell(input_size, hidden_size)
            # init.orthogonal(lstm.weight_ih)
            # init.orthogonal(lstm.weight_hh)
            # init.constant(lstm.bias_ih, 1.0)
            # init.constant(lstm.bias_hh, 1.0)
            self.layers.append(lstm)
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.encoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout, bidirectional=config.bidirec)

        self.config = config
        # self.posenc = PositionalEncoding(config)
        self.hidden_size = config.encoder_hidden_size
        self.sigmoid = nn.Sigmoid()
        self.layers = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.in1 = nn.InstanceNorm1d(config.decoder_hidden_size, eps=1e-10)
        self.in2 = nn.InstanceNorm1d(config.decoder_hidden_size, eps=1e-10)

        self.conv1 = nn.Conv1d(config.decoder_hidden_size, config.decoder_hidden_size, kernel_size=3, padding=0, dilation=1)
        self.selu1 = nn.Sequential(nn.SELU(), nn.AlphaDropout(p=0.05))
        self.conv2 = nn.Conv1d(config.decoder_hidden_size, config.decoder_hidden_size, kernel_size=3, padding=0, dilation=2)
        self.selu2 = nn.Sequential(nn.SELU(), nn.AlphaDropout(p=0.05))
        self.conv3 = nn.Conv1d(config.decoder_hidden_size, config.decoder_hidden_size, kernel_size=3, padding=0, dilation=3)
        self.selu3 = nn.Sequential(nn.SELU(), nn.AlphaDropout(p=0.05))

    def forward(self, input, lengths, cnn=True):
        embs = pack(self.embedding(input), lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        if not self.config.bidirec:
            return outputs, state
        else:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
            state = (state[0][1::2], state[1][1::2])
            o_ = outputs
            if cnn:
                outputs = outputs.transpose(0,1).transpose(1,2)
                outputs = self.selu1(self.conv1(outputs))
                outputs = self.selu2(self.conv2(outputs))
                outputs = self.selu3(self.conv3(outputs))
                conv = outputs.transpose(1,2).transpose(0,1)
                # outputs = self.sigmoid(outputs) * o_
                outputs = o_

            return outputs, conv, state


class gated_rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(gated_rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.encoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)
        self.gated = nn.Sequential(nn.Linear(config.encoder_hidden_size, 1), nn.Sigmoid())

    def forward(self, input, lengths):
        embs = pack(self.embedding(input), lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        p = self.gated(outputs)
        outputs = outputs * p
        return outputs, state


class rnn_decoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None, score_fn=None):
        super(rnn_decoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = StackedLSTM(input_size=config.emb_size, hidden_size=config.decoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)
        self.score_fn = score_fn
        if self.score_fn.startswith('general'):
            self.linear = nn.Linear(config.decoder_hidden_size, config.emb_size)
        elif score_fn.startswith('concat'):
            self.linear_query = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)
            self.linear_weight = nn.Linear(config.emb_size, config.decoder_hidden_size)
            self.linear_v = nn.Linear(config.decoder_hidden_size, 1)
        elif not self.score_fn.startswith('dot'):
            self.linear = nn.Linear(config.decoder_hidden_size, vocab_size)

        if hasattr(config, 'att_act'):
            activation = config.att_act
            print('use attention activation %s' % activation)
        else:
            activation = None

        self.attention = models.global_attention(config.decoder_hidden_size, dropout=config.dropout, softmax=True)
        self.attention_conv = models.global_attention(config.decoder_hidden_size, dropout=config.dropout)
        self.attention_in = models.global_attention(config.decoder_hidden_size, dropout=config.dropout)
        self.hidden_size = config.decoder_hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config


    def forward(self, inputs, init_state, contexts, contexts_conv):
        embs = self.embedding(inputs)
        outputs, state, attns = [], init_state, []
        for emb in embs.split(1):
            output, state = self.rnn(emb.squeeze(0), state)
            output_conv, attn_weights = self.attention_conv(output, contexts_conv, conv=True)
            output, attn_weights = self.attention(output_conv, contexts)

            output = output + output_conv
            outputs += [output]
            attns += [attn_weights]
        outputs = torch.stack(outputs)
        attns = torch.stack(attns)

        return outputs, state

    def compute_score(self, hiddens):
        if self.score_fn.startswith('general'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(self.linear(hiddens), Variable(self.embedding.weight.t().data))
            else:
                scores = torch.matmul(self.linear(hiddens), self.embedding.weight.t())
        elif self.score_fn.startswith('concat'):
            if self.score_fn.endswith('not'):
                scores = self.linear_v(torch.tanh(self.linear_query(hiddens).unsqueeze(1) + \
                                      self.linear_weight(Variable(self.embedding.weight.data)).unsqueeze(0))).squeeze(2)
            else:
                scores = self.linear_v(torch.tanh(self.linear_query(hiddens).unsqueeze(1) + \
                                      self.linear_weight(self.embedding.weight).unsqueeze(0))).squeeze(2)
        elif self.score_fn.startswith('dot'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(hiddens, Variable(self.embedding.weight.t().data))
            else:
                scores = torch.matmul(hiddens, self.embedding.weight.t())
        else:
            scores = self.linear(hiddens)
        return scores

    def sample(self, input, init_state, contexts, contexts_conv):
        inputs, outputs, sample_ids, state = [], [], [], init_state
        attns = []
        inputs += input
        max_time_step = self.config.max_tgt_len

        for i in range(max_time_step):
            output, state, attn_weights = self.sample_one(inputs[i], state, contexts, contexts_conv)
            predicted = output.max(1)[1]
            inputs += [predicted]
            sample_ids += [predicted]
            outputs += [output]
            attns += [attn_weights]

        sample_ids = torch.stack(sample_ids)
        # attns = torch.stack(attns)
        return sample_ids, (outputs, attns)

    def sample_one(self, input, state, contexts, contexts_conv):
        emb = self.embedding(input)
        output, state = self.rnn(emb, state)

        hidden_conv, attn_weights = self.attention_conv(output, contexts_conv, conv=True)
        hidden, attn_weights = self.attention(hidden_conv, contexts)

        hidden = hidden + hidden_conv
        output = self.compute_score(hidden)

        return output, state, attn_weights
