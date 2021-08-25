## PyTorch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim


import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch_geometric
from torch_geometric.utils import to_dense_adj
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data



DATASET_PATH = "../data"

CHECKPOINT_PATH = "../saved_models/tutorial7"

# Setting the seed
pl.seed_everything(123)

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

"""
Pretrained Layers to be used.
"""
gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

"""
Loading datasets from torch geometric
"""
Pubmed = torch_geometric.datasets.Planetoid(root='/', name='Pubmed')
data_pub = Pubmed[0]

# Fetch the Dataset object
dataset = torch_geometric.datasets.Planetoid(root='/', name='Cora')

# One graph is one Data object, with the following attributes:
#   edge_index:     the adjacency list of shape [2, num_edges] (COO format)
#   x:              the feature matrix of shape [num_nodes, num_features]
#   y:              node labels of shape [num_nodes]
#   train_mask:     a boolean mask of shape [num_nodes], indicating the train set
#                   (similarly for val_mask and test_mask)
data_cora = dataset[0]


cit = torch_geometric.datasets.Planetoid(root='/', name='Citeseer')
data_cit = cit[0]

#######Training, computing accuracy and loss##############
class NodeLevelGNN(pl.LightningModule):
    
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        
        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()
        
    
    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)
        
        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, "Unknown forward mode: %s" % mode
        
        loss = self.loss_module(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc
        
        
    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        return optimizer
        
        
    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss)#####Automatic logging from torch lightning Module
        self.log('train_acc', acc)
        return loss
        
        
    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc)
        
        
    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)