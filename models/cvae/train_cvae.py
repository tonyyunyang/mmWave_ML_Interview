import cv2
import numpy as np
import torch
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import imageio


def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False


def train_for_one_epoch(epoch_idx, model, data_loader, optimizer, crtierion, config, device, save_path):
    r"""
    Method to run the training for one epoch.
    :param epoch_idx: iteration number of current epoch
    :param model: VAE model
    :param mnist_loader: Data loder for mnist
    :param optimizer: optimzier to be used taken from config
    :param crtierion: For computing the loss
    :param config: configuration for the current run
    :return:
    """
    recon_losses = []
    kl_losses = []
    losses = []
    # We ignore the label for VAE
    for im, im_name in tqdm(data_loader):
        # Handle case where dataloader still returns a tuple but we only want first element
        if isinstance(im, (tuple, list)):
            im = im[0]
        im = im.float().to(device)
        
        optimizer.zero_grad()
        # Remove label parameter from model call
        output = model(im)
        mean = output['mean']
        
        # if check_nan(mean, "mean"): continue

        std, log_variance = None, None
        if config['model_params']['log_variance']:
            log_variance = output['log_variance']
        else:
            std = output['std']
        generated_im = output['image']
        generated_im = torch.clamp(generated_im, 0, 1)  # Clamp values between 0 and 1
        if config['train_params']['save_training_image']:
            # Handle full batch at once
            batch_size = im.size(0)
            
            # Convert tensors to numpy arrays for full batch
            input_imgs = (255.0 * im.detach()).cpu().numpy()[:, 0].astype(np.uint8)
            output_imgs = (255.0 * generated_im.detach()).cpu().numpy()[:, 0].astype(np.uint8)
            
            # Handle each image in batch
            for idx in range(batch_size):
                current_name = im_name[idx] if isinstance(im_name, (list, tuple)) else im_name
                
                # Create sample-specific directory
                sample_dir_input = os.path.join(save_path, current_name, "input")
                sample_dir_output = os.path.join(save_path, current_name, "output")
                os.makedirs(sample_dir_input, exist_ok=True)
                os.makedirs(sample_dir_output, exist_ok=True)
                
                # Save input and output images with epoch number
                input_path = os.path.join(sample_dir_input, f"input_epoch{epoch_idx}.jpg")
                output_path = os.path.join(sample_dir_output, f"output_epoch{epoch_idx}.jpg")
                
                # Save as JPEG
                Image.fromarray(input_imgs[idx]).save(input_path)
                Image.fromarray(output_imgs[idx]).save(output_path)
        
        if config['model_params']['log_variance']:
            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(log_variance) + mean ** 2 - 1 - log_variance, dim=-1))
        else:
            kl_loss = torch.mean(0.5 * torch.sum(std ** 2 + mean ** 2 - 1 - torch.log(std ** 2), dim=-1))
        recon_loss = crtierion(generated_im, im)
        loss = recon_loss + config['train_params']['kl_weight'] * kl_loss
        recon_losses.append(recon_loss.item())
        losses.append(loss.item())
        kl_losses.append(kl_loss.item())
        loss.backward()
        optimizer.step()
    print('Finished epoch: {} | Recon Loss : {:.4f} | KL Loss : {:.4f}'.format(epoch_idx + 1,
                                                                               np.mean(recon_losses),
                                                                               np.mean(kl_losses)))
    return np.mean(losses)


def visualize_latent_space(config, model, data_loader, save_fig_path, device):
    r"""
    Method to visualize the latent dimension by simply plotting the means for each of the images
    :param config: Config file used to create the model
    :param model:
    :param data_loader:
    :param save_fig_path: Path where the latent space image will be saved
    :return:
    """
    means = []
    
    # Modified to unpack only images from dataloader
    for im, im_name in tqdm(data_loader):
        # Handle case where dataloader still returns a tuple but we only want first element
        if isinstance(im, (tuple, list)):
            im = im[0]
        im = im.float().to(device)
        # Remove label parameter from model call
        output = model(im)
        mean = output['mean']
        means.append(mean)
    
    means = torch.cat(means, dim=0)
    if model.latent_dim != 2:
        print('Latent dimension > 2 and hence projecting')
        U, _, V = torch.pca_lowrank(means, center=True, niter=2)
        proj_means = torch.matmul(means, V[:, :2])
        if not os.path.exists(config['train_params']['task_name']):
            os.mkdir(config['train_params']['task_name'])
        pickle.dump(V, open('{}/pca_matrix.pkl'.format(config['train_params']['task_name']), 'wb'))
        means = proj_means
    
    # Modified plotting code to show points without color-coding by label
    fig, ax = plt.subplots()
    ax.scatter(means[:, 0].cpu().numpy(), means[:, 1].cpu().numpy(), s=10, alpha=0.5)
    ax.grid(True)
    plt.savefig(save_fig_path)