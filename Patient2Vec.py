"""
Patient2Vec: a self-attentive representation learning framework
author: Jinghe Zhang
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


class Patient2Vec(nn.Module):
    """
    Self-attentive representation learning framework,
    including convolutional embedding layer,
    recurrent autoencoder with an encoder, recurrent module, and a decoder.
    In addition, a linear layer is on top of each decode step and the weights are shared at these step.
    """

    def __init__(self, input_size, hidden_size, n_layers, att_dim, initrange,
                 output_size, rnn_type, seq_len, pad_size, n_filters, bi, dropout_p=0.5):
        """
        Initilize a recurrent model
        :param input_size: int
        :param hidden_size: int
        :param n_layers: number of layers; int
        :param att_dim: dimension of the attention; int
        :param initrange: upper bound of the initial weights; symmetric
        :param output_size: int
        :param rnn_type: str, such as 'GRU'
        :param seq_len: length of the sequence; int
        :param pad_size: padding size; int
        :param n_filters: number of hops; int
        :param bi: bidirectional; bool
        :param dropout_p: dropout rate; float
        """
        super(Patient2Vec, self).__init__()

        self.initrange = initrange
        # convolution
        self.b = 1
        if bi:
            self.b = 2

        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=input_size, stride=2)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=hidden_size * self.b, stride=2)
        # Bidirectional RNN
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, n_layers, dropout=dropout_p,
                                         batch_first=True, bias=True, bidirectional=bi)
        # initialize 2-layer attention weight matrics
        self.att_w1 = nn.Linear(hidden_size * self.b, att_dim, bias=False)
        # final linear layer
        self.linear = nn.Linear(hidden_size * self.b * n_filters + 3, output_size, bias=True)

        self.func_softmax = nn.Softmax()
        self.func_sigmoid = nn.Sigmoid()
        self.func_tanh = nn.Hardtanh(0, 1)
        # Add dropout
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.init_weights()

        self.pad_size = pad_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.n_filters = n_filters

    def init_weights(self):
        """
        weight initialization
        """
        for param in self.parameters():
            param.data.uniform_(-self.initrange, self.initrange)

    def convolutional_layer(self, inputs):
        convolution_all = []
        conv_wts = []
        for i in range(self.seq_len):
            convolution_one_month = []
            for j in range(self.pad_size):
                convolution = self.conv(torch.unsqueeze(inputs[:, i, j], dim=1))
                convolution_one_month.append(convolution)
            convolution_one_month = torch.stack(convolution_one_month)
            convolution_one_month = torch.squeeze(convolution_one_month, dim=3)
            convolution_one_month = torch.transpose(convolution_one_month, 0, 1)
            convolution_one_month = torch.transpose(convolution_one_month, 1, 2)
            convolution_one_month = torch.squeeze(convolution_one_month, dim=1)
            convolution_one_month = self.func_tanh(convolution_one_month)
            convolution_one_month = torch.unsqueeze(convolution_one_month, dim=1)
            vec = torch.bmm(convolution_one_month, inputs[:, i])
            convolution_all.append(vec)
            conv_wts.append(convolution_one_month)
        convolution_all = torch.stack(convolution_all, dim=1)
        convolution_all = torch.squeeze(convolution_all, dim=2)
        conv_wts = torch.squeeze(torch.stack(conv_wts, dim=1), dim=2)
        return convolution_all, conv_wts

    def encode_rnn(self, embedding, batch_size):
        self.weight = next(self.parameters()).data
        init_state = (Variable(self.weight.new(self.n_layers * self.b, batch_size, self.hidden_size).zero_()))
        embedding = self.dropout(embedding)
        outputs_rnn, states_rnn = self.rnn(embedding, init_state)
        return outputs_rnn

    def add_beta_attention(self, states, batch_size):
        # beta attention
        att_wts = []
        for i in range(self.seq_len):
            m1 = self.conv2(torch.unsqueeze(states[:, i], dim=1))
            att_wts.append(torch.squeeze(m1, dim=2))
        att_wts = torch.stack(att_wts, dim=2)
        att_beta = []
        for i in range(self.n_filters):
            a0 = self.func_softmax(att_wts[:, i])
            att_beta.append(a0)
        att_beta = torch.stack(att_beta, dim=1)
        context = torch.bmm(att_beta, states)
        context = context.view(batch_size, -1)
        return att_beta, context

    def forward(self, inputs, inputs_other, batch_size):
        """
        the recurrent module
        """
        # Convolutional
        convolutions, alpha = self.convolutional_layer(inputs)
        # RNN
        states_rnn = self.encode_rnn(convolutions, batch_size)
        # Add attentions and get context vector
        beta, context = self.add_beta_attention(states_rnn, batch_size)
        # Final linear layer with demographic info added as extra variables
        context_v2 = torch.cat((context, inputs_other), 1)
        linear_y = self.linear(context_v2)
        out = self.func_softmax(linear_y)
        return out, alpha, beta


def get_loss(pred, y, criterion, mtr, a=0.5):
    """
    To calculate loss
    :param pred: predicted value
    :param y: actual value
    :param criterion: nn.CrossEntropyLoss
    :param mtr: beta matrix
    """
    mtr_t = torch.transpose(mtr, 1, 2)
    aa = torch.bmm(mtr, mtr_t)
    loss_fn = 0
    for i in range(aa.size()[0]):
        aai = torch.add(aa[i, ], Variable(torch.neg(torch.eye(mtr.size()[1]))))
        loss_fn += torch.trace(torch.mul(aai, aai).data)
    loss_fn /= aa.size()[0]
    loss = torch.add(criterion(pred, y), Variable(torch.FloatTensor([loss_fn * a])))
    return loss
