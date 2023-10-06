import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 config
                 ):
        super(Encoder, self).__init__()
        activation_map = {
            'relu': nn.ReLU(),
            'leaky': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        
        self.config = config
        
        ##### Validate the configuration for the model is correctly setup #######
        assert config['conv_activation_fn'] is None or config['conv_activation_fn'] in activation_map
        self.latent_dim = config['latent_dim']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Encoder is just Conv bn activation blocks
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config['convbn_channels'][i], config['convbn_channels'][i + 1],
                          kernel_size=config['conv_kernel_size'][i], stride=config['conv_kernel_strides'][i],padding=1),
                nn.BatchNorm2d(config['convbn_channels'][i + 1]),
                activation_map[config['conv_activation_fn']],
            )
            for i in range(config['convbn_blocks']-1)
        ])
        
        enc_last_idx = config['convbn_blocks']
        self.encoder_layers.append(
            nn.Sequential(
                nn.Conv2d(config['convbn_channels'][enc_last_idx - 1], config['convbn_channels'][enc_last_idx],
                          kernel_size=config['conv_kernel_size'][enc_last_idx-1],
                          stride=config['conv_kernel_strides'][enc_last_idx-1], padding=1),
            )
        )
    
    def forward(self, x):
        out = x
        for layer in self.encoder_layers:
            out = layer(out)
        return out


def get_encoder(config):
    encoder = Encoder(
        config=config['model_params']
    )
    return encoder



