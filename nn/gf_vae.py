# Import pytorch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class VAE(nn.Module):
    def __init__(self, input_dim=1024, inter_dim=512, latent_dim=128):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, inter_dim),
            nn.BatchNorm1d(inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, latent_dim * 2)
        )

        self.decoder =  nn.Sequential(
            nn.Linear(latent_dim, inter_dim),
            nn.BatchNorm1d(inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, input_dim)
        )

        self.kl = 0

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        # reshape data
        org_size = x.size()
        batch = org_size[0]
        x = x.view(batch, -1)

        #encode
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)

        self.kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

        z = self.reparameterise(mu, logvar)
        recon_x = self.decoder(z).view(size=org_size)

        return recon_x, z

if __name__ == "__main__":
    ## Sanity Check
    model = VAE()
    
    # GPU check                
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
        print("Run on GPU...")
    else:
        print("Run on CPU...")

    # Generator Definition  
    model = model.to(device)
    summary(model, (1024,))

    # Test forward pass
    z = torch.randn(5,1024)
    z = z.to(device)
    out = model(z)
    # Check output shape
    assert(out.detach().cpu().numpy().shape == (5,1024))
    print("Forward pass successful")