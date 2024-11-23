import cv2
import numpy as np
import torch
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt


def train_for_one_epoch(epoch_idx, model, mnist_loader, optimizer, crtierion, config, device):
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
    for im, label in tqdm(mnist_loader):
        im = im.float().to(device)
        label = label.long().to(device)
        optimizer.zero_grad()
        output = model(im, label)
        mean = output['mean']
        std, log_variance = None, None
        if config['model_params']['log_variance']:
            log_variance = output['log_variance']
        else:
            std = output['std']
        generated_im = output['image']
        if config['train_params']['save_training_image']:
            cv2.imwrite('input.jpeg', (255 * (im.detach() + 1) / 2).cpu().numpy()[0, 0])
            cv2.imwrite('output.jpeg', (255 * (generated_im.detach() + 1) / 2).cpu().numpy()[0, 0])
        
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
    labels = []
    means = []
    
    for im, label in tqdm(data_loader):
        im = im.float().to(device)
        label = label.long().to(device)
        output = model(im, label)
        labels.append(label)
        mean = output['mean']
        means.append(mean)
    
    labels = torch.cat(labels, dim=0).reshape(-1)
    means = torch.cat(means, dim=0)
    if model.latent_dim != 2:
        print('Latent dimension > 2 and hence projecting')
        U, _, V = torch.pca_lowrank(means, center=True, niter=2)
        proj_means = torch.matmul(means, V[:, :2])
        if not os.path.exists(config['train_params']['task_name']):
            os.mkdir(config['train_params']['task_name'])
        pickle.dump(V, open('{}/pca_matrix.pkl'.format(config['train_params']['task_name']), 'wb'))
        means = proj_means
    
    fig, ax = plt.subplots()
    for num in range(10):
        idxs = torch.where(labels == num)[0]
        ax.scatter(means[idxs, 0].cpu().numpy(), means[idxs, 1].cpu().numpy(), s=10, label=str(num),
                   alpha=1.0, edgecolors='none')
    ax.legend()
    ax.grid(True)
    plt.savefig(save_fig_path)