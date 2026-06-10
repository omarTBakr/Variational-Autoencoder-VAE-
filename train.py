import os
import numpy as np
import torch
from PIL import Image

from data import CelebData, get_data
from modeling import VAE

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCHES = 128
EPOCHS  = 30


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────

def run_epoch(model: VAE, loader, optimizer=None) -> tuple[float, float]:
    """
    Run one full pass over `loader`.
    - If `optimizer` is given  → training mode (gradients updated).
    - If `optimizer` is None   → eval mode   (no gradients).

    Returns (avg_recon_loss, avg_total_loss) normalised per sample.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_recon = 0.0
    total_loss  = 0.0
    n_samples   = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, _ in loader:
            batch = x.to(device)

            if is_train:
                optimizer.zero_grad()

            recon_batch, mu, logvar = model(batch)
            loss, recon, _ = model.vae_loss(batch, recon_batch, mu, logvar)

            if is_train:
                loss.backward()
                optimizer.step()

            total_recon += recon.item()
            total_loss  += loss.item()
            n_samples   += batch.size(0)

    return total_recon / n_samples, total_loss / n_samples


# ─────────────────────────────────────────────
# generation
# ─────────────────────────────────────────────

def generate_nsamples(model: VAE, samples: int) -> None:
    os.makedirs("samples", exist_ok=True)
    # BUG FIX: use model.latent_size instead of hardcoded 200
    with torch.no_grad():
        for i in range(samples):
            z = torch.randn(1, model.latent_size).to(device)
            img = model.generate(z)
            img = img.cpu().numpy()[0]
            img = (img * 255).astype(np.uint8).transpose(1, 2, 0)
            Image.fromarray(img).save(f"samples/{i}.png")
    print(f"Saved {samples} generated images → samples/")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    data: CelebData = get_data(batch_size=BATCHES, img_size=64)

    model     = VAE(latent_size=200).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    print(f"Training on {device}\n{'─'*55}")
    print(f"{'Epoch':>6} | {'Train Recon':>12} | {'Train Total':>12} | {'Val Recon':>10} | {'Val Total':>10}")
    print(f"{'─'*55}")

    for epoch in range(1, EPOCHS + 1):
        train_recon, train_total = run_epoch(model, data.train_loader, optimizer)
        val_recon,   val_total   = run_epoch(model, data.val_loader)

        print(
            f"{epoch:>6}/{EPOCHS} | "
            f"{train_recon:>12.4f} | "
            f"{train_total:>12.4f} | "
            f"{val_recon:>10.4f} | "
            f"{val_total:>10.4f}"
        )

    # ── Final evaluation on the test set ──────
    print(f"\n{'─'*55}")
    print("Final evaluation on test set …")
    test_recon, test_total = run_epoch(model, data.test_loader)
    print(f"  Reconstruction loss : {test_recon:.4f}")
    print(f"  Total ELBO loss     : {test_total:.4f}")
    print(f"{'─'*55}\n")

    # ── Generate sample images ─────────────────
    generate_nsamples(model, 10)


if __name__ == "__main__":
    main()