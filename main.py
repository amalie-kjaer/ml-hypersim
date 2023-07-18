import numpy as np
from tqdm import tqdm
from model import GCN
from dataset import HypersimDataset
from torch_geometric.loader import DataLoader
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad() 
        out,_ = model(data.x.float(), data.edge_index.type(torch.long), data.batch.type(torch.long))
        loss = criterion(out, data.y.type(torch.long))
        loss.backward()
        optimizer.step()

def test(model, loader):
    model.eval()
    correct = 0
    embeddings = []

    for data in tqdm(loader):
        # out, embedding = model(data.x.float(), data.edge_index.type(torch.long), data.batch.type(torch.long))
        # embeddings.append(embedding)
        out, _= model(data.x.float(), data.edge_index.type(torch.long), data.batch.type(torch.long))
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    print(out[0,:])
    return correct / len(loader.dataset)

def balanced_train_dataloader(batch_size=64, n_samples_per_class=5):
    class_labels = np.array(np.arange(6)) # TODO get value from HypersimDataset class
    class_indices_dict = {label: [] for label in class_labels}

    print('fetching class labels...')
    for idx in tqdm(range(len(dataset.graphs))):
        label = dataset[idx].y.item()
        class_indices_dict[label].append(idx)

    selected_indices = []
    for class_indices in class_indices_dict.values():
        selected_class_indices = random.sample(class_indices, n_samples_per_class)
        selected_indices.extend(selected_class_indices)

    train_dataset = torch.utils.data.Subset(dataset, selected_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return train_dataset, train_loader

if __name__ == "__main__":
    dataset = HypersimDataset(r'C:\Users\amali\Documents\ds_research\ml-hypersim')
    # train_dataset = dataset[:2000]
    # test_dataset = dataset[60000:]
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

    

    model = GCN(feature_size=train_dataset[0].x.shape[1], hidden_channels=64)
    model = model.to(device)
    
    for epoch in range(0, 200):
        print('Training') 
        train(model)
        print('Calculating accuracies')
        train_acc = test(model, train_loader)
        # test_acc = test(test_loader)
        # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')