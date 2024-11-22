import torch
import torch.nn as nn


class CVAE(nn.Module):
    def __init__(self, config):

        super(CVAE, self).__init__()

        activation_map = {
            'relu': nn.ReLU(),
            'leaky': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        
        self.config = config

        assert config['transpose_activation_fn'] is None or config['transpose_activation_fn'] in activation_map
        assert config['dec_fc_activation_fn'] is None or config['dec_fc_activation_fn'] in activation_map
        assert config['conv_activation_fn'] is None or config['conv_activation_fn'] in activation_map
        assert config['enc_fc_activation_fn'] is None or config['enc_fc_activation_fn'] in activation_map
        assert config['enc_fc_layers'][-1] == config['dec_fc_layers'][0] == config['latent_dim'], "Latent dimension must be same as fc layers number"
        
        self.transposebn_channels = config['transposebn_channels']
        self.latent_dim = config['latent_dim']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Encoder is just Conv BatchNorm blocks followed by fc for mean and variance
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config['convbn_channels'][i], config['convbn_channels'][i + 1],
                          kernel_size=config['conv_kernel_size'][i], stride=config['conv_kernel_strides'][i]),
                nn.BatchNorm2d(config['convbn_channels'][i + 1]),
                activation_map[config['conv_activation_fn']]
            )
            for i in range(config['convbn_blocks'])
        ])
        
        encoder_mu_activation = nn.Identity() if config['enc_fc_mu_activation'] is None else activation_map[
            config['enc_fc_mu_activation']]
        self.encoder_mu_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config['enc_fc_layers'][i], config['enc_fc_layers'][i + 1]),
                encoder_mu_activation
            )
            for i in range(len(config['enc_fc_layers']) - 1)
        ])

        encoder_var_activation = nn.Identity() if config['enc_fc_var_activation'] is None else activation_map[
            config['enc_fc_var_activation']]
        self.encoder_var_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config['enc_fc_layers'][i], config['enc_fc_layers'][i + 1]),
                encoder_var_activation
            )
            for i in range(len(config['enc_fc_layers']) - 1)
        ])

        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(config['transposebn_channels'][i], config['transposebn_channels'][i + 1],
                                   kernel_size=config['transpose_kernel_size'][i],
                                   stride=config['transpose_kernel_strides'][i]),
                nn.BatchNorm2d(config['transposebn_channels'][i + 1]),
                activation_map[config['transpose_activation_fn']]
            )
            for i in range(config['transpose_bn_blocks'])
        ])
        
        self.decoder_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config['dec_fc_layers'][i], config['dec_fc_layers'][i + 1]),
                activation_map[config['dec_fc_activation_fn']]
            )
            for i in range(len(config['dec_fc_layers']) - 1)
        
        ])
    
    def forward(self, x, label=None):
        out = x
            
        for layer in self.encoder_layers:
            out = layer(out)
        out = out.reshape((x.size(0), -1))
        mu = out
        for layer in self.encoder_mu_fc:
            mu = layer(mu)
        std = out
        for layer in self.encoder_var_fc:
            std = layer(std)

        z = self.reparameterize(mu, std)
        generated_out = self.generate(z, label)
        if self.config['log_variance']:
            return {
                'mean': mu,
                'log_variance': std,
                'image': generated_out,
            }
        else:
            return {
                'mean': mu,
                'std': std,
                'image': generated_out,
            }
    
    def generate(self, z, label=None):
        out = z
        for layer in self.decoder_fc:
            out = layer(out)
        # Figure out how to reshape based on desired number of channels in transpose convolution
        hw = torch.as_tensor(out.size(-1) / self.transposebn_channels[0]).to(self.device)
        spatial = int(torch.sqrt(hw))
        assert spatial * spatial == hw
        out = out.reshape((z.size(0), -1, spatial, spatial))
        for layer in self.decoder_layers:
            out = layer(out)
        return out
    
    def reparameterize(self, mu, std_or_logvariance):
        if self.config['log_variance']:
            std = torch.exp(0.5 * std_or_logvariance)
        else:
            std = std_or_logvariance
        z = torch.randn_like(std)
        return z * std + mu


def get_model(config):
    model = CVAE(
        config=config['model_params']
    )
    return model


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import yaml
    
    config_path = '../config/vae_kl_latent4.yaml'
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    model = get_model(config)
    labels = torch.zeros((3)).long()
    labels[0] = 0
    labels[1] = 2
    out = model(torch.rand((3,1,28,28)), labels)
    print(out['mean'].shape)
    print(out['image'].shape)

