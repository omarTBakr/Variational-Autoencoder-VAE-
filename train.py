import os
import numpy as np
import torch
from PIL import Image

from data import CelebData, get_data
from modeling import VAE

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCHES = 16 
EPOCHS = 30 

def main():
 
    data:CelebData = get_data(batch_size=BATCHES, img_size=64)

    # MODEL
    model = VAE(latent_size=200).to(device)

    # OPTIMIZER
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # TRAINING
    for epoch in range(EPOCHS):
        for x, _ in data.train_loader:
            batch = x.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = model.vae_loss(batch, recon_batch, mu, logvar)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")



    def generate_nsamples(samples :int) ->None:
        os.makedirs("samples", exist_ok=True)
        for i in range(samples):
            img = model.generate(torch.randn(1, 200).to(device))
            img = img.detach().cpu().numpy()[0]
            img = (img * 255).astype(np.uint8)
            img = img.transpose(1, 2, 0)
            img = Image.fromarray(img)
            img.save(f"samples/{i}.png")
    generate_nsamples(10)
if __name__ == "__main__":
    main()