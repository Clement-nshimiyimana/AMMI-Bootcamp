GCN: SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS
Paper Reference: https://arxiv.org/abs/1609.02907

Install the requirement libraries
!pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
!pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
!pip install -q torch-geometric
Clone the repository
!git clone https://github.com/Clement-nshimiyimana/AMMI-Bootcamp.git
Train the model of Cora dataset
!python GCN/Main.py  --num-epochs 200 --data 'Cora' --num-hid  16  
Results
Dataset	Type	Nodes	Edges	Classes	Features	Test Accuracy
Cora	Citation Network	2,708	5,429	7	1,433	81.5 士 1
Citeseer	Citation Network	3,327	4,732	6	3,703	69 士 0.8