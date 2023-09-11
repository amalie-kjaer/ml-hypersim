import torch
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool

class CustomGCN(torch.nn.Module):
    def __init__(self, feature_size, hidden_channels, num_layers, dropout_rate):
        super(CustomGCN, self).__init__()
        # torch.manual_seed(12345)
        self.dropout_rate = dropout_rate
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(feature_size, hidden_channels))
            feature_size = hidden_channels
        self.lin = torch.nn.Linear(hidden_channels, 6)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()
        
        x = global_mean_pool(x, batch)
        embedding = x
        
        x = F.dropout(x, p = self.dropout_rate, training=self.training)
        x = self.lin(x)
        
        # return x, embedding # CHANGE BACK
        return embedding, F.log_softmax(x, dim=1)

class GIN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(1, hidden_channels),
                       BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.lin1 = Linear(hidden_channels*3, hidden_channels*3)
        self.lin2 = Linear(hidden_channels*3, 6)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h, F.log_softmax(h, dim=1)