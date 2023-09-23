import torch
from torch import nn
import math

def f(x):
    return 2 * torch.sin(x) + x ** 0.8
n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5  )

X = torch.ones((2,4,6))
Y = torch.ones((2,1,6))
Z = torch.bmm(X,Y)
Z.shape #torch.Size([2,1,6 ])

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim = -1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:

        else:
            return None 

class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        self.W_k = nn.Linear(key_size, num_hiddens, bias = False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias = False)
        self.W_v = nn.Linear(num_hiddens, 1, bias = False)
        self.dropout == nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        #the dim of queris is [batch size, number of queries, queries length]
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1) #broadcast
        features = torch.tanh(features)
        scores = self.W_v(features).squeeze(-1)
        #the dim of scores now is [batch size, number of queries, number of keys]
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights). values) #bmm: matrix multiplication with a batch size
    
class DotProduceAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProduceAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    
    def foward(self, queries, keys ,values, valid_lens = None)
        d= queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights). values)



