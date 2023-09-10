import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, feature_size, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(feature_size, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 6) # no. possible output labels

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        embedding = x

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x, embedding

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
        
        return x, embedding