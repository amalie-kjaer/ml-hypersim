import numpy as np
import os
import random
import json
import yaml
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from dataset import HypersimDataset
from model import *
from utils import *

def load_dataset_full():
    dataset = HypersimDataset(root='./')
    return dataset

def load_dataset_small(config):
    file = './graphs.json'
    classes = [int(x) for x in config['dataset']['classes'].split(',')]
    if os.path.isfile(file):
        with open(file, 'r') as f:
            graphs = json.load(f)
        dataset = HypersimDataset(root='./', classes=classes, presaved_graphs=graphs)
    else: 
        dataset = HypersimDataset(root='./', classes=classes)
        graphs = dataset.graphs
        with open('./graphs.json', 'w') as f:
            json.dump(graphs, f, indent=2)
    return dataset

def load_dataset_small_balanced(config):
    dataset_small = load_dataset_small(config)
    
    # Construct dict of keys: class label, values: idx
    # (Ensures equal number of random samples in each class of dataset_small
    # i.e. creates a balanced dataset.)
    file = './class_indices_dict.json'
    if  os.path.isfile(file):
        with open(file, 'r') as f:
            class_indices_dict = json.load(f)
    else:
        classes = [int(x) for x in config['dataset']['classes'].split(',')]
        new_class_indices = np.array(np.arange(len(classes)), dtype=int)
        class_indices_dict = {int(label): [] for label in new_class_indices}
        for idx in tqdm(range(len(dataset_small))):
            label = dataset_small[idx].y.item()
            class_indices_dict[int(label)].append(idx)
    
    # Randomly select {samples_per_class} samples from each class of dataset_small
    random_indices = []
    samples_per_class = int(config['dataset']['samples_per_class'])
    for class_indices in class_indices_dict.values():
        random_indices.extend(random.sample(class_indices, samples_per_class))
    
    # Create small, balanced dataset from random indices
    dataset = torch.utils.data.Subset(dataset_small, random_indices)
    return dataset

def train_one_epoch(train_loader, model, optimizer, criterion):
    model.train()
    for data in train_loader: 
        optimizer.zero_grad()
        out, _ = model(data.x.float(), data.edge_index.type(torch.long), data.batch.type(torch.long))
        loss = criterion(out, data.y.type(torch.long))
        loss.backward()
        optimizer.step()

def test_model(loader, model):
    model.eval()
    correct = 0
    
    for data in loader:
        out, _ = model(data.x.float(), data.edge_index.type(torch.long), data.batch.type(torch.long))  
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

def train_model():
    config = load_config()

    # Initialize wandb
    if eval(config['wandb']['log']) == True:
        wandb.init(project=config['wandb']['project_name'], name=config['wandb']['run_name'])
        wandb.config.update(config)

    # Load dataset specified in config
    train_dataset_type = config['dataset']['dataset_type']
    if train_dataset_type == 'dataset_full':
        train_dataset = load_dataset_full()
    elif train_dataset_type == 'dataset_small':
        train_dataset = load_dataset_small(config)
    elif train_dataset_type == 'dataset_small_balanced':
        train_dataset = load_dataset_small_balanced(config)
    
    train_loader = DataLoader(train_dataset, batch_size=int(config['datamodule']['batch_size']), shuffle=True, drop_last=False)

    model_name = config['model']['model_name']
    model = globals()[model_name](feature_size=1, hidden_channels=int(config['model']['hidden_channels'])) #TODO change feature_size

    optimizer_name = config['model']['optimizer']
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=float(config['model']['lr']))

    criterion_name = config['model']['criterion']
    criterion = getattr(nn, criterion_name)()

    num_epochs = int(config['trainer']['epochs'])
    # Train model for {num_epochs} epochs
    for epoch in range (num_epochs):
        train_one_epoch(train_loader, model, optimizer, criterion)
        train_acc = test_model(train_loader, model)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')

        # Logging to wandb
        if eval(config['wandb']['log']) == True:
            wandb.log({"Train Accuracy": train_acc})
    
    print("done")
    wandb.finish()