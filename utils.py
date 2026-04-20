import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def generate_distributed_datasets(k: int, alpha: float, save_dir: str):
    
    #Generates k subsets of FashionMNIST using Dirichlet distribution
    #and saves them to save_dir.
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST normalization
    ])
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    targets = np.array(dataset.targets)
    
    # Get class indices
    num_classes = len(np.unique(targets))
    idx_by_class = {c: np.where(targets == c)[0] for c in range(num_classes)}
    
    # Dirichlet allocation
    client_indices = [[] for _ in range(k)]
    proportions = np.random.dirichlet([alpha] * k, size=num_classes)

    for c in range(num_classes):
        idxs = idx_by_class[c]
        np.random.shuffle(idxs)
        split_pos = np.round(np.cumsum(proportions[c] * len(idxs))).astype(int)
        splits = np.split(idxs, split_pos[:-1])
        for i in range(k):
            client_indices[i].extend(splits[i])

    # Save datasets
    os.makedirs(save_dir, exist_ok=True)
    for i in range(k):
        client_data = [dataset[j] for j in client_indices[i]]
        with open(os.path.join(save_dir, f'client_{i}.pkl'), 'wb') as f:
            pickle.dump(client_data, f)


# This should be outside of the above function
def load_client_data(cid: int, data_dir: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    Loads and returns train and test DataLoaders for a given client ID.
    """
    with open(os.path.join(data_dir, f'client_{cid}.pkl'), 'rb') as f:
        data = pickle.load(f)

    # Convert list of (image, label) tuples into tensors
    images = torch.stack([d[0] for d in data])
    labels = torch.tensor([d[1] for d in data])

    # Split into train and validation
    train_size = int(0.8 * len(images))
    val_size = len(images) - train_size
    train_dataset, val_dataset = random_split(list(zip(images, labels)), [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

