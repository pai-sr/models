import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base.base_model import BaseModel

class LSTMCell(BaseModel):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, x, hidden):
        hx, cx = hidden
        hx = hx.to(torch.float32)
        cx = cx.to(torch.float32)
        x = x.view(-1, x.size(1)).to(torch.float32)

        gates = self.x2h(x) + self.h2h(hx)
        #gates = gates.squeeze() # shape 안맞음
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, F.tanh(cy))

        return (hy, cy)

class LSTMModel(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.cell_list = nn.ModuleList()

        self.cell_list.append(LSTMCell(self.input_size,
                                           self.hidden_size,
                                           self.bias))
        for l in range(1, self.num_layers):
            self.cell_list.append(LSTMCell(self.hidden_size,
                                               self.hidden_size,
                                               self.bias))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hx=None):
        if hx is None:
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float64))
        else:
            h0 = hx

        outs = []
        hn = []
        cn = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))

        for t in range(x.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.cell_list[layer](
                        x[:, t, :],
                        (hidden[layer][0], hidden[layer][1])
                    )
                else:
                    hidden_l = self.cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                    )
                hidden[layer] = hidden_l
            outs.append(hidden_l[0])
            hn.append(hidden_l[0])
            cn.append(hidden_l[1])

        hn = torch.stack(hn)
        cn = torch.stack(cn)
        outs = torch.stack(outs)

        return outs, (hn, cn)


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, bias=True):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = LSTMModel(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias, output_size=num_classes)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        out = self.fc(output[-1:, :, :])
        out = out.to(torch.float64)
        return out