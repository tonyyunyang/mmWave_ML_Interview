import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

def train(model, train_loader, config, device):
    ckpt_dir = config['train_params']['task_name']
    os.makedirs(ckpt_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['train_params']['lr'],
        weight_decay=config['train_params']['weight_decay']
    )
    
    best_loss = float('inf')
    
    for epoch in range(config['train_params']['epochs']):
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["train_params"]["epochs"]}') as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader)
        
        # Save checkpoint if loss improved
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(ckpt_dir, config['train_params']['ckpt_name']))
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config["train_params"]["epochs"]}], Loss: {epoch_loss:.4f}')