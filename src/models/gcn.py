import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    
    def __init__(self, dim_in, dim_h, dim_out):
      super().__init__()
      self.gcn1 = GCNConv(dim_in, dim_h)
      self.gcn2 = GCNConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.5, training=self.training)
        h = self.gcn1(h, edge_index).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn2(h, edge_index)
        return h, F.log_softmax(h, dim=1)


# class GraphConvolution(nn.Module):
#     def __init__(self, in_dim, out_dim, support, activation = None, featureless = False, dropout=0.0, bias=False):
#         super(GraphConvolution, self).__init__()
        
#         self.support = support
#         self.featureless = featureless

#         for i in range(len(self.support)):
#             setattr(self, 'weight{}'.format(i), nn.Parameter(torch.randn(in_dim, out_dim)))
        
#         if bias:
#             self.b = nn.Parameter(torch.zeros(1, out_dim))
        
#         self.activation = activation
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, input):
#         x = self.dropout(input)

#         for i in range(len(self.support)):
#             if self.featureless:
#                 pre_sup = getattr(self, 'weight{}'.format(i))
#             else:
#                 pre_sup = x.mm(getattr(self, 'weight{}'.format(i)))
            
#             if i == 0:
#                 out = self.support[i].mm(pre_sup)
#             else:
#                 out = out + self.support[i].mm(pre_sup)
        
#         if self.activation is not None:
#             out = self.activation(out)
        
#         self.embedding = out

#         return out


# class GCN(nn.Module):
#     def __init__(self, input_dim, support, dropout, nclass):
#         super(GCN, self).__init__()
        
#         self.gc1 = GraphConvolution(input_dim, 128, support, activation=nn.ReLU(), featureless=True, dropout=dropout)
#         self.gc2 = GraphConvolution(128, nclass, support, dropout=dropout)

#     def forward(self, x):
#         x = self.gc1(x)
#         x = self.gc2(x)
#         return x

#     def get_embedding(self):
#         return self.gc1.embedding


