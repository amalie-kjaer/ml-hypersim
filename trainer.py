import numpy as np
import os
import random
import json
import yaml
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from dataset import HypersimDataset
from model import *
from utils import *
from sklearn.metrics import confusion_matrix

def load_dataset_full():
    dataset = HypersimDataset(root='./')
    return dataset

def load_dataset_small(config):
    file = './checkpoints/graphs.json'
    classes = [int(x) for x in config['dataset']['classes'].split(',')]
    if os.path.isfile(file):
        with open(file, 'r') as f:
            graphs = json.load(f)
        dataset = HypersimDataset(root='./', classes=classes, presaved_graphs=graphs)
    else: 
        dataset = HypersimDataset(root='./', classes=classes)
        graphs = dataset.graphs
        with open('./checkpoints/graphs.json', 'w') as f:
            json.dump(graphs, f, indent=2)
    return dataset

def load_dataset_small_balanced(config):
    dataset_small = load_dataset_small(config)
    
    # Construct dict of keys: class label, values: idx
    # (Ensures equal number of random samples in each class of dataset_small
    # i.e. creates a balanced dataset.)
    file = './checkpoints/class_indices_dict.json'
    if  os.path.isfile(file):
        with open(file, 'r') as f:
            class_indices_dict = json.load(f)
    else:
        classes = [int(x) for x in config['dataset']['classes'].split(',')]
        new_class_indices = np.array(np.arange(len(classes)), dtype=int)
        class_indices_dict = {int(label): [] for label in new_class_indices}
        for idx in tqdm(range(len(dataset_small))):
            label = dataset_small[idx][0].y.item()
            class_indices_dict[int(label)].append(idx)
        with open('./checkpoints/class_indices_dict.json', 'w') as f:
            json.dump(class_indices_dict, f, indent=2)
    
    # Randomly select {samples_per_class} samples from each class of dataset_small
    random_indices = []
    samples_per_class = int(config['dataset']['samples_per_class'])
    for class_indices in class_indices_dict.values():
        random_indices.extend(random.sample(class_indices, samples_per_class))
    
    # Create small, balanced dataset from random indices
    # dataset = torch.utils.data.Subset(dataset_small, random_indices)
    dataset = dataset_small[random_indices]
    return dataset, random_indices

def train_one_epoch(train_loader, model, optimizer, criterion, config):
    model.train()
    for data in train_loader: 
        optimizer.zero_grad()
        out, _ = model(data[0].x.float(), data[0].edge_index.type(torch.long), data[0].batch.type(torch.long))
        loss = criterion(out, data[0].y.type(torch.long))
        loss.backward()
        optimizer.step()
    if eval(config['wandb']['log']) == True:
        wandb.log({"Loss": loss})

def test_model(loader, model):
    model.eval()
    correct = 0
    true_labels = []
    predicted_labels = []
    shuffled_idx = []

    for data in loader:
        out, _ = model(data[0].x.float(), data[0].edge_index.type(torch.long), data[0].batch.type(torch.long))  
        pred = out.argmax(dim=1)
        correct += int((pred == data[0].y).sum())
        true_labels.extend(data[0].y.tolist())
        predicted_labels.extend(pred.tolist())
        shuffled_idx.extend(data[1].tolist())
    return correct / len(loader.dataset), true_labels, predicted_labels, shuffled_idx

def train_model():
    config = load_config()

    # Load dataset specified in config
    dataset_type = config['dataset']['dataset_type']
    if dataset_type == 'dataset_full':
        dataset = load_dataset_full()
    elif dataset_type == 'dataset_small':
        dataset = load_dataset_small(config)
    elif dataset_type == 'dataset_small_balanced':
        dataset, random_indices = load_dataset_small_balanced(config)
    
    # Split dataset into train and test
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])
    
    train_loader = DataLoader(train_dataset, batch_size=int(config['datamodule']['batch_size']), shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=int(config['datamodule']['batch_size']), shuffle=False, drop_last=False)

    model_name = config['model']['model_name']
    model = globals()[model_name](feature_size=1, hidden_channels=int(config['model']['hidden_channels']),  num_layers=int(config['model']['num_layers']),
                                  dropout_rate=float(config['model']['dropout'])) #TODO change feature_size

    optimizer_name = config['model']['optimizer']
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=float(config['model']['lr']))

    criterion_name = config['model']['criterion']
    criterion = getattr(nn, criterion_name)()

    num_epochs = int(config['trainer']['epochs'])
    
    # Initialize wandb
    if eval(config['wandb']['log']) == True:
        wandb.init(project=config['wandb']['project_name'], name=config['wandb']['run_name'])
        wandb.config.update(config)

        # Log input data to wandb
        # if dataset_type == 'dataset_small_balanced':
        #     input_images = []
        #     dataset_small = load_dataset_small(config)
        #     print("Visalizing input images...")
        #     for idx in random_indices:
        #         caption=f"Label: {dataset_small[idx][0].y.item()}"
        #         image_wandb = log_visualization_wandb(dataset_small, idx, caption)
        #         input_images.append(image_wandb)
        #     wandb.log({"Input images": input_images})

    # Train model for {num_epochs} epochs
    for epoch in range (num_epochs):
        train_one_epoch(train_loader, model, optimizer, criterion, config)
        train_acc, true_labels, predicted_labels, _ = test_model(train_loader, model)
        test_acc, true_labels_test, predicted_labels_test, _ = test_model(test_loader, model)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        # Log to wandb
        if eval(config['wandb']['log']) == True:

            wandb.log({"Train Accuracy": train_acc})
            wandb.log({"Test Accuracy": test_acc})

            # For last epoch, save lables and plot confusion matrix
            if epoch == num_epochs-1:
                print("Constructing confusion matrix...")
                train_acc, true_labels, predicted_labels, shuffled_idx = test_model(train_loader, model)
                class_names = ['Bathroom', 'Bedroom', 'Kitchen', 'Living room', 'Office', 'Restaurant']
                wandb.log({"Confusion Matrix (Training)": wandb.plot.confusion_matrix(probs=None, y_true=true_labels, preds=predicted_labels, class_names=class_names)})
                wandb.log({"Confusion Matrix (Testing)": wandb.plot.confusion_matrix(probs=None, y_true=true_labels_test, preds=predicted_labels_test, class_names=class_names)})
                
                # Plot incorrect results:
                # wrong_images = []
                # correct_images = []
                # for i in range(len(true_labels)):
                #     if true_labels[i] != predicted_labels[i]:
                #         caption=f"Predicted: {class_names[predicted_labels[i]]}, Actual: {class_names[true_labels[i]]}"
                #         image_wandb_wrong = log_visualization_wandb(dataset_small, shuffled_idx[i], caption)
                #         wrong_images.append(image_wandb_wrong)
                #     else:
                #         caption=f"Predicted: {class_names[predicted_labels[i]]}, Actual: {class_names[true_labels[i]]}"
                #         image_wandb_correct = log_visualization_wandb(dataset_small, shuffled_idx[i], caption)
                #         correct_images.append(image_wandb_correct)
                # print('Logging wrong images...')
                # wandb.log({"Wrong images": wrong_images})
                # print('Logging correct images...')
                # wandb.log({"Correct images": correct_images})

    if eval(config['wandb']['log']) == True:
        wandb.finish()

    print("done")