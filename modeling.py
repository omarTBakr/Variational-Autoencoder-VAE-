import torch
from torch import nn 

class VAE(nn.Module):
    def __init__(self, latent_size: int):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(

            nn.Conv2d(in_channels=3 , out_channels=32 , kernel_size=4 , stride=2 , padding=1) , 
            nn.BatchNorm2d(num_features=32) , 
            nn.ReLU(True) , 
            nn.Dropout(p=0.2) , 
            nn.Conv2d(in_channels=32 , out_channels=64 , kernel_size=4 , stride=2 , padding=1) , 
            nn.BatchNorm2d(num_features=64) ,
            nn.ReLU(True) , 
            nn.Dropout(p=0.2) , 
            nn.Conv2d(in_channels=64 , out_channels=128 , kernel_size=4 , stride=2 , padding=1) , 
            nn.BatchNorm2d(num_features=128) , 
            nn.ReLU(True) , 
            nn.Dropout(p=0.2) , 
            nn.Conv2d(in_channels=128 , out_channels=256 , kernel_size=4 , stride=2 , padding=1) , 
            nn.BatchNorm2d(num_features=256) , 
            nn.ReLU(True) , 
            nn.Dropout(p=0.2) ,
            nn.Conv2d(in_channels=256 , out_channels=512 , kernel_size=4 , stride=2 , padding=1) , 
            nn.BatchNorm2d(num_features=512) , 
            nn.ReLU(True) , 
            nn.Dropout(p=0.2) ,
            nn.Flatten() , 
             
        )
        self.fc_mu = nn.Linear(512 * 2 * 2 , latent_size)
        self.fc_logvar = nn.Linear(512 * 2 * 2 , latent_size)
        self.decoder = nn.Sequential(

            # project latent vector back to spatial feature map
            nn.Linear(in_features=latent_size, out_features=512 * 2 * 2),
            nn.ReLU(True),  # BUG FIX: activation was missing after Linear projection
            nn.Unflatten(dim=1, unflattened_size=(512, 2, 2)),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        # Clamp logvar to prevent exp() overflow → NaN/Inf during training
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def vae_loss(self, x, recon_x, mu, logvar):
        """Returns (total_loss, recon_loss, kl_loss) as separate tensors."""
        # Clamp logvar here too for the KL computation
        logvar = torch.clamp(logvar, min=-10, max=10)
        # recon = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        recon = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + kl, recon, kl
    
    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
    def generate(self, z):
        """Generate images from latent vectors. Always runs in eval mode."""
        self.eval()
        with torch.no_grad():
            return self.decoder(z)
        
            




if __name__ == "__main__":
    model = VAE(latent_size=200)
    print(model)


