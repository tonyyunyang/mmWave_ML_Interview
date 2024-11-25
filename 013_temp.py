import torch
import numpy as np
import torch.nn as nn
import os
import yaml
import joblib
from models.cnn.cnn import CNN
from models.cvae.cvae import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Randomize a new sample
new_sample = np.random.uniform(0, 1, (256, 256))
new_sample = torch.FloatTensor(new_sample).unsqueeze(0)  # Add batch dimension\


def inference_pipeline(sample, cvae_config, cnn_config, kmeans_model):    
    # Initialize models
    cvae_model = get_model(cvae_config['model_params']).to(device)
    cnn_model = CNN(cnn_config['model_params']).to(device)

    cvae_weights_path = os.path.join(cvae_config['train_params']['task_name'], cvae_config['train_params']['ckpt_name'])
    cnn_weights_path = os.path.join(cnn_config['train_params']['task_name'], cnn_config['train_params']['ckpt_name'])

    # Load pretrained weights
    cvae_model.load_state_dict(torch.load(cvae_weights_path, map_location=device))
    cnn_model.load_state_dict(torch.load(cnn_weights_path, map_location=device))
    
    # Ensure models are in eval mode
    cvae_model.eval()
    cnn_model.eval()
    
    # Add channel dimension if needed
    if len(sample.shape) == 3:
        sample = sample.unsqueeze(1)
    
    with torch.no_grad():
        # Get activity cluster from CVAE embeddings
        cvae_output = cvae_model(sample.to(device))
        if isinstance(cvae_output, dict):
            features = cvae_output['mean']  # Get latent mean vector
        else:
            features = cvae_output
        activity_cluster = kmeans_model.predict(features.cpu().numpy())[0]
        
        # Get subject prediction from CNN
        cnn_output = cnn_model(sample.to(device))
        subject_idx = torch.argmax(cnn_output, dim=1).item()
        subject_label = chr(subject_idx + ord('A'))  # Convert back to letter
        
    return activity_cluster, subject_label


cvae_config = yaml.load(open('config/config.yaml', 'r'), Loader=yaml.FullLoader)
cnn_config = yaml.load(open('config/cnn_config.yaml', 'r'), Loader=yaml.FullLoader)
kmeans_model = joblib.load(os.path.join(cvae_config['train_params']['task_name'], 'kmeans.joblib'))

