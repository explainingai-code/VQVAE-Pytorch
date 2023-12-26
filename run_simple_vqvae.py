import torch
import cv2
import torchvision
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from einops import rearrange
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        
        self.pre_quant_conv = nn.Conv2d(4, 2, kernel_size=1)
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=2)
        self.post_quant_conv = nn.Conv2d(2, 4, kernel_size=1)
        
        # Commitment Loss Beta
        self.beta = 0.2
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        
        
    def forward(self, x):
        # B, C, H, W
        encoded_output = self.encoder(x)
        quant_input = self.pre_quant_conv(encoded_output)
        
        ## Quantization
        B, C, H, W = quant_input.shape
        quant_input = quant_input.permute(0, 2, 3, 1)
        quant_input = quant_input.reshape((quant_input.size(0), -1, quant_input.size(-1)))
        
        # Compute pairwise distances
        dist = torch.cdist(quant_input, self.embedding.weight[None, :].repeat((quant_input.size(0), 1, 1)))
        
        # Find index of nearest embedding
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        # Select the embedding weights
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        quant_input = quant_input.reshape((-1, quant_input.size(-1)))
        
        # Compute losses
        commitment_loss = torch.mean((quant_out.detach() - quant_input)**2)
        codebook_loss = torch.mean((quant_out - quant_input.detach())**2)
        quantize_losses = codebook_loss + self.beta*commitment_loss
        
        # Ensure straight through gradient
        quant_out = quant_input + (quant_out - quant_input).detach()
        
        # Reshaping back to original input shape
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        
        
        ## Decoder part
        decoder_input = self.post_quant_conv(quant_out)
        output = self.decoder(decoder_input)
        return output, quantize_losses
    
def train_vqvae():
    mnist = MnistDataset('train', im_path='data/train/images')
    mnist_test = MnistDataset('test', im_path='data/test/images')
    mnist_loader = DataLoader(mnist, batch_size=64, shuffle=True, num_workers=4)
    
    model = VQVAE().to(device)
    
    num_epochs = 20
    optimizer = Adam(model.parameters(), lr=1E-3)
    criterion = torch.nn.MSELoss()
    
    for epoch_idx in range(num_epochs):
        for im, label in tqdm(mnist_loader):
            im = im.float().to(device)
            optimizer.zero_grad()
            out, quantize_loss = model(im)
            
            recon_loss = criterion(out, im)
            loss = recon_loss + quantize_loss
            loss.backward()
            optimizer.step()
        print('Finished epoch {}'.format(epoch_idx+1))
    print('Done Training...')
    
    # Reconstruction part
    
    idxs = torch.randint(0, len(mnist_test), (100, ))
    ims = torch.cat([mnist_test[idx][0][None, :] for idx in idxs]).float()
    ims = ims.to(device)
    model.eval()
    
    
    generated_im, _ = model(ims)
    ims = (ims+1)/2
    generated_im = 1 - (generated_im+1)/2
    out = torch.hstack([ims, generated_im])
    output = rearrange(out, 'b c h w -> b () h (c w)')
    grid = torchvision.utils.make_grid(output.detach().cpu(), nrow=10)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save('reconstruction.png')
    
    print('Done Reconstruction ...')
    
if __name__ == '__main__':
    train_vqvae()
    
    
    
    
    
    
    
    
    
    
    
