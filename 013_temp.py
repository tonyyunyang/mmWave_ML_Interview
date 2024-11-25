import torch
import numpy as np
import torch.nn as nn
from models.cnn.cnn import CNN
from models.cvae.cvae import get_model


# Randomize a new sample
new_sample = np.random.uniform(0, 1, (256, 256))
new_sample = torch.FloatTensor(new_sample).unsqueeze(0)  # Add batch dimension\


def predict_both(sample, cvae_model, cnn_model, kmeans, device):
    # Cluster prediction
    cvae_model.eval()
    with torch.no_grad():
        features = cvae_model(sample.to(device))
        cluster = kmeans.predict(features.cpu().numpy())[0]
        
        # Identity prediction
        identity = cnn_model(sample.to(device))
        pred_identity = torch.argmax(identity, dim=1).item()
    
    return cluster, pred_identity

