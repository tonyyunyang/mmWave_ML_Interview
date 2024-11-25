import time
import torch
import os
import yaml
import joblib
import numpy as np
from tqdm import tqdm
from datetime import datetime

from models.cnn.cnn import CNN
from models.cvae.cvae import CVAE

class InferencePipeline:
    def __init__(self, cvae_config_path, cnn_config_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configurations
        self.cvae_config = yaml.load(open(cvae_config_path, 'r'), Loader=yaml.FullLoader)
        self.cnn_config = yaml.load(open(cnn_config_path, 'r'), Loader=yaml.FullLoader)
        
        # Initialize models
        self.cvae_model = CVAE(self.cvae_config['model_params']).to(self.device)
        self.cnn_model = CNN(self.cnn_config['model_params']).to(self.device)
        
        # Load model weights
        self._load_model_weights()
        
        # Load KMeans model
        kmeans_path = os.path.join(self.cvae_config['train_params']['task_name'], 'kmeans.joblib')
        self.kmeans_model = joblib.load(kmeans_path)
        
        # Set models to eval mode
        self.cvae_model.eval()
        self.cnn_model.eval()

    def _load_model_weights(self):
        cvae_weights_path = os.path.join(
            self.cvae_config['train_params']['task_name'], 
            self.cvae_config['train_params']['ckpt_name']
        )
        cnn_weights_path = os.path.join(
            self.cnn_config['train_params']['task_name'], 
            self.cnn_config['train_params']['ckpt_name']
        )
        
        self.cvae_model.load_state_dict(torch.load(cvae_weights_path, map_location=self.device))
        
        cnn_checkpoint = torch.load(cnn_weights_path, map_location=self.device)
        self.cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
        self.cnn_model.to(self.device)

    def predict_single(self, sample):
        if len(sample.shape) == 3:
            sample = sample.unsqueeze(1)
            
        with torch.no_grad():
            cvae_output = self.cvae_model(sample.to(self.device))
            features = cvae_output['mean'] if isinstance(cvae_output, dict) else cvae_output
            activity_cluster = self.kmeans_model.predict(features.cpu().numpy())[0]
            
            # Get subject prediction
            cnn_output = self.cnn_model(sample.to(self.device))
            subject_idx = torch.argmax(cnn_output, dim=1).item()
            subject_label = chr(subject_idx + ord('A'))
            
        return activity_cluster, subject_label

    def predict_batch(self, samples, batch_size=32, show_progress=True):
        if len(samples.shape) == 3:
            samples = samples.unsqueeze(1)
            
        n_samples = len(samples)
        activity_clusters = []
        subject_labels = []
        
        iterator = range(0, n_samples, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing samples")
            
        with torch.no_grad():
            for i in iterator:
                batch = samples[i:i + batch_size]
                
                # Get activity clusters
                cvae_output = self.cvae_model(batch.to(self.device))
                features = cvae_output['mean'] if isinstance(cvae_output, dict) else cvae_output
                batch_clusters = self.kmeans_model.predict(features.cpu().numpy())
                
                # Get subject predictions
                cnn_output = self.cnn_model(batch.to(self.device))
                subject_indices = torch.argmax(cnn_output, dim=1).cpu().numpy()
                batch_subjects = [chr(idx + ord('A')) for idx in subject_indices]
                
                activity_clusters.extend(batch_clusters)
                subject_labels.extend(batch_subjects)
                
        return activity_clusters, subject_labels


def real_time_prediction(pipeline, samples, num_samples=100, delay=0.5):
    print(f"\nStarting real-time prediction simulation at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Will process {num_samples} samples with {delay} second delay between each\n")
    
    num_samples = min(num_samples, len(samples))
    
    try:
        for i in range(num_samples):
            current_sample = samples[i]
            
            start_time = time.time()
            activity_cluster, subject_label = pipeline.predict_single(current_sample)
            prediction_time = time.time() - start_time
            
            print(f"Sample {i+1}/{num_samples}")
            print(f"├── Activity Cluster: {activity_cluster}")
            print(f"├── Subject: {subject_label}")
            print(f"└── Prediction Time: {prediction_time:.3f}s")
            print()
            
            if i < num_samples - 1: 
                time.sleep(delay)
                
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user")
    


def generate_samples(num_samples=100, height=64, width=64, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    samples = []
    for _ in range(num_samples):
        sample = np.random.rand(height, width)
        
        center_y, center_x = height//2, width//2
        y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
        mask = x*x + y*y <= (min(height, width)//4)**2
        sample[mask] = np.random.rand()
        
        # Normalize to [0, 1]
        sample = (sample - sample.min()) / (sample.max() - sample.min())
        
        # Convert to tensor and add channel dimension
        sample = torch.FloatTensor(sample).unsqueeze(0)
        samples.append(sample)
    
    return torch.stack(samples)