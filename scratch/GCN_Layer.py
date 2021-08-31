import torch
import math
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(torch.nn.Module):
    """
    GCN layer
    """
    def __init__(self, in_features, out_features, bias= True):

        super().__init__()

        # dimension of the input features and output features


        self.in_features = in_features
        self.out_features = out_features

        self.weight = None

        # initialise the weight matrix

        stdv = 1 / math.sqrt(self.out_features)

        self.weight = torch.FloatTensor(in_features, out_features).uniform_(-stdv, stdv)

        self.weight = nn.Parameter(self.weight, requires_grad=True)

        if bias:
            self.bias = torch.FloatTensor(out_features).uniform_(-stdv, stdv)
            self.bias = nn.Parameter(self.bias, requires_grad=True)
        else:
            self.bias = None


    def forward(self, fts, adj):

        A_ = adj.to(device) + torch.eye(adj.shape[0]).to(device)
        D_power_  = torch.diag(torch.pow(A_.sum(dim=-1),-0.5))
        support = torch.mm(A_, D_power_)
        support = torch.mm(D_power_, support)

        output = torch.mm(fts, self.weight)
    
        output = torch.sparse.mm(support, output)

        if self.bias is not None:
            return output + self.bias
        else:
            return output