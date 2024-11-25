from torch.utils.data import Dataset
import einops
import torch


class SignatureFigDataset(Dataset):
    def __init__(self, signature_figs, filenames):
        self.signature_figs = signature_figs
        self.filenames = filenames

    def __len__(self):
        return len(self.signature_figs)

    def __getitem__(self, idx):
        signature_fig = self.signature_figs[idx] # (H, W, C)
        signature_fig = einops.rearrange(signature_fig, "h w c -> c h w")
        return signature_fig / 255.0, self.filenames[idx]


class IdentityDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor([ord(label) - ord('A') for label in labels])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx] / 255.0
        label = self.labels[idx]
            
        return image, label

