from tqdm import tqdm
from model import GCN
from dataset import HypersimDataset
from torch_geometric.loader import DataLoader
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():      
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad() 
        out,_ = model(data.x.float(), data.edge_index.type(torch.long), data.batch.type(torch.long))
        loss = criterion(out, data.y.type(torch.long))
        loss.backward()
        optimizer.step()

def test(loader):
    # model.eval()
    correct = 0
    embeddings = []

    for data in tqdm(loader):
        # out, embedding = model(data.x.float(), data.edge_index.type(torch.long), data.batch.type(torch.long))
        # embeddings.append(embedding)
        out, _= model(data.x.float(), data.edge_index.type(torch.long), data.batch.type(torch.long))
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct / len(loader.dataset)


if __name__ == "__main__":
    dataset = HypersimDataset(r'C:\Users\amali\Documents\ds_research\ml-hypersim')
    train_dataset = dataset[:2000]
    test_dataset = dataset[60000:]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)
    
    model = GCN(feature_size=train_dataset[0].x.shape[1], hidden_channels=64)
    model = model.to(device)
    
    for epoch in range(0, 50):
        print('Training')
        
        model.train()
        train()
        print('Calculating accuracies')
        
        model.eval()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')