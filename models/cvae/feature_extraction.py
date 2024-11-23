import torch
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from collections import defaultdict


def extract_features(model, dataloader, device):
    model.eval()
    features = []
    filenames_list = []
    
    with torch.no_grad():
        for batch, filenames in dataloader:
            batch = batch.float().to(device)
            features_batch = encode_features(model, batch)
            features.append(features_batch.cpu().numpy())
            filenames_list.extend(filenames)
    
    features = np.concatenate(features, axis=0)
    return features, filenames_list


def encode_features(model, x):
    out = x
    for layer in model.encoder_layers:
        out = layer(out)
    
    out = out.reshape((x.size(0), -1))
    
    mu = out.clone()
    std = out.clone()
    
    for layer in model.encoder_mu_fc:
        mu = layer(mu)
    for layer in model.encoder_var_fc:
        std = layer(std)
    
    z = model.reparameterize(mu, std)
    
    return z