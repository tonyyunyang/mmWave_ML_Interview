from torch.utils.data import Dataset
import einops


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