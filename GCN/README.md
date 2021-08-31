Graph Convolutional Networks

This is pytorch implementation of paper :https://arxiv.org/pdf/1609.02907.pdf which was implemented before in tensorFlow. 
The main idea of this paper is to use Graph Convolutional Networks for the task of (semi-supervised) classification of nodes in a graph.


Author: Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)

Installation

python setup.py install
Requirements
Pytorch (>0.12)


Run the demo


cd GCN

python Main.py

In this work, we load data from pytorch geometric.  We used citation network data (Cora, Citeseer and Pubmed). The original datasets can be found here: http://www.cs.umd.edu/~sen/lbc-proj/LBC.html. In paper's version (see data folder) they used dataset splits provided by https://github.com/kimiyoung/planetoid (Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov, Revisiting Semi-Supervised Learning with Graph Embeddings, ICML 2016).
