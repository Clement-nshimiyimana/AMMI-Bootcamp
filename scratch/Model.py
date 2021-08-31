import torch 
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from GCN_Layer import GraphConvolution

class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, fts, adj):
        fts = F.relu(self.gc1(fts, adj))
        fts = F.dropout(fts, self.dropout, training=self.training)
        fts = self.gc2(fts, adj)
        return F.log_softmax(fts, dim=1)