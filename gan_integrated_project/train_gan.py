import argparse
import math
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from gan_model import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SECRET_BITS,
    MODEL_VERSION,
    DenseDecoder,
    DenseEncoder,
    Discriminator,
    pack_message,
    tile_bits,
    unpack_message,
)

# Robust path resolution independent of terminal working directory
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "cover_images"
DEFAULT_CHECKPOINT_DIR = BASE_DIR / "checkpoints"


class CoverDataset(Dataset):
    """Loads cover images with optional data augmentation and canonical resizing."""

    def __init__(self, folder: Path, image_size: int = DEFAULT_IMAGE_SIZE, augment: bool = True):
        self.paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            self.paths.extend(Path(folder).glob(ext))
        if not self.paths:
            raise RuntimeError(
                f"No images found in {folder}. Please add PNG or JPG images to the folder first."
            )

        tf_list = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        if augment:
            tf_list.append(transforms.RandomHorizontalFlip(p=0.5))
        tf_list.append(transforms.ToTensor())
        self.tf = transforms.Compose(tf_list)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def evaluate_model(
    encoder: nn.Module,
    decoder: nn.Module,
    val_loader: DataLoader,
    device: str,
    secret_bits: int,
    image_size: int,
) -> tuple[float, float, bool]:
    """
    Evaluates the model on unseen cover images with fresh secret bits.
    Simulates uint8 quantization (PNG export/import) to ensure real-world extraction works.
    """
    encoder.eval()
    decoder.eval()
    total_acc = 0.0
    total_psnr = 0.0
    count = 0
    all_roundtrips_passed = True

    test_messages = ["Hello World!", "Test Secret 123", "Stego 2026!"]

    with torch.no_grad():
        for i, cover in enumerate(val_loader):
            cover = cover.to(device)
            b, _, h, w = cover.shape

            # Random bits accuracy
            bits = torch.randint(0, 2, (b, secret_bits), device=device).float()
            msg_map = tile_bits(bits, h, w, secret_bits)
            stego = encoder(cover, msg_map)

            # Simulated uint8 quantization
            stego_quant = torch.clamp((stego * 255.0).round() / 255.0, 0.0, 1.0)
            logits = decoder(stego_quant)

            # Bit accuracy
            pred_bits = (logits > 0).float()
            batch_acc = (pred_bits == msg_map).float().mean().item()

            # PSNR
            mse = F.mse_loss(stego_quant, cover).item()
            psnr = -10.0 * math.log10(max(mse, 1e-10))

            total_acc += batch_acc * b
            total_psnr += psnr * b
            count += b

            # Test text message round-trip on first batch
            if i == 0:
                for msg in test_messages:
                    try:
                        p_bits = torch.from_numpy(pack_message(msg, secret_bits)).unsqueeze(0).to(device)
                        p_map = tile_bits(p_bits, h, w, secret_bits)
                        s = encoder(cover[0:1], p_map)
                        s_uint8 = torch.clamp((s * 255.0).round() / 255.0, 0.0, 1.0)
                        pred_log = decoder(s_uint8)
                        rec = unpack_message(pred_log.flatten(), secret_bits)
                        if rec != msg:
                            all_roundtrips_passed = False
                    except Exception:
                        all_roundtrips_passed = False

    encoder.train()
    decoder.train()
    avg_acc = total_acc / max(count, 1)
    avg_psnr = total_psnr / max(count, 1)
    return avg_acc, avg_psnr, all_roundtrips_passed


def main():
    parser = argparse.ArgumentParser(description="Train GAN for Image Steganography")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Directory containing cover images")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs (default: 35)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training (default: 4)")
    parser.add_argument("--lr", type=float, default=0.001, help="Generator learning rate (default: 0.001)")
    parser.add_argument("--lr_d", type=float, default=0.0002, help="Discriminator learning rate (default: 0.0002)")
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE, help="Image resolution (default: 128)")
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE, help="Channels in hidden conv layers (default: 48)")
    parser.add_argument("--secret_bits", type=int, default=DEFAULT_SECRET_BITS, help="Secret bit payload size (default: 256)")
    parser.add_argument("--mse_weight", type=float, default=15.0, help="Weight for image MSE distortion loss (default: 15.0)")
    parser.add_argument("--adv_weight", type=float, default=0.05, help="Weight for adversarial loss (default: 0.05)")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("GAN-Based Image Steganography - Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Image Resolution: {args.image_size}x{args.image_size}")
    print(f"Secret Payload: {args.secret_bits} bits ({args.secret_bits // 8 - 4} UTF-8 bytes)")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | Generator LR: {args.lr}")
    print(f"Data directory: {args.data_dir}")
    print("=" * 70)

    # Load dataset
    full_dataset = CoverDataset(Path(args.data_dir), image_size=args.image_size, augment=True)
    num_total = len(full_dataset)
    print(f"Found {num_total} cover images in dataset.")

    # Split into train (80%) and validation (20%)
    num_val = max(1, int(num_total * 0.2))
    num_train = num_total - num_val
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [num_train, num_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"Training on {num_train} images, validating on {num_val} images.")
    if num_total < 50:
        print(
            "Note: Training with 25 images is intended for educational demonstration on CPU.\n"
            "Data augmentation and spatial tiling are used to maximize generalization."
        )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=min(args.batch_size, num_val), shuffle=False)

    # Instantiate networks
    encoder = DenseEncoder(data_depth=1, hidden_size=args.hidden_size).to(device)
    decoder = DenseDecoder(data_depth=1, hidden_size=args.hidden_size).to(device)
    discriminator = Discriminator(hidden_size=32).to(device)

    opt_g = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d)
    bce_loss = nn.BCEWithLogitsLoss()

    DEFAULT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    final_psnr = 0.0

    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        discriminator.train()

        sum_ml, sum_il, sum_dl, sum_acc = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for cover in train_loader:
            cover = cover.to(device)
            b, _, h, w = cover.shape

            # Generate random secret bits and spatial message map
            bits = torch.randint(0, 2, (b, args.secret_bits), device=device).float()
            msg_map = tile_bits(bits, h, w, args.secret_bits)

            # ----------------------------------------------------
            # 1. Train Discriminator
            # ----------------------------------------------------
            with torch.no_grad():
                fake = encoder(cover, msg_map)

            real_logits = discriminator(cover)
            fake_logits = discriminator(fake.detach())
            dl = bce_loss(real_logits, torch.ones_like(real_logits)) + bce_loss(
                fake_logits, torch.zeros_like(fake_logits)
            )

            opt_d.zero_grad()
            dl.backward()
            opt_d.step()

            # ----------------------------------------------------
            # 2. Train Generator (Encoder + Decoder)
            # ----------------------------------------------------
            fake = encoder(cover, msg_map)

            # Inject simulated uint8 quantization noise during training
            # to make the decoder immune to PNG saving/loading quantization
            q_noise = (torch.rand_like(fake) - 0.5) / 255.0
            pred_map = decoder(fake + q_noise)

            ml = F.binary_cross_entropy_with_logits(pred_map, msg_map)
            il = F.mse_loss(fake, cover)
            adv = bce_loss(discriminator(fake), torch.ones_like(discriminator(fake)))

            gl = ml + args.mse_weight * il + args.adv_weight * adv

            opt_g.zero_grad()
            gl.backward()
            opt_g.step()

            batch_bit_acc = ((pred_map > 0).float() == msg_map).float().mean().item()
            sum_ml += ml.item()
            sum_il += il.item()
            sum_dl += dl.item()
            sum_acc += batch_bit_acc
            n_batches += 1

        avg_ml = sum_ml / n_batches
        avg_il = sum_il / n_batches
        avg_dl = sum_dl / n_batches
        avg_train_acc = sum_acc / n_batches
        epoch_psnr = -10.0 * math.log10(max(avg_il, 1e-10))

        # Evaluate every 5 epochs or on the last epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            val_acc, val_psnr, roundtrip_ok = evaluate_model(
                encoder, decoder, val_loader, device, args.secret_bits, args.image_size
            )
            print(
                f"Epoch {epoch + 1:2d}/{args.epochs:2d} | "
                f"msg_BCE: {avg_ml:.4f} | img_MSE: {avg_il:.6f} | "
                f"train_acc: {avg_train_acc * 100:.2f}% | val_acc: {val_acc * 100:.2f}% | "
                f"PSNR: {epoch_psnr:.1f}dB | D_loss: {avg_dl:.4f} | "
                f"Roundtrip: {'PASS' if roundtrip_ok else 'RETRY'}"
            )
            final_psnr = epoch_psnr
            best_val_acc = val_acc
        else:
            print(
                f"Epoch {epoch + 1:2d}/{args.epochs:2d} | "
                f"msg_BCE: {avg_ml:.4f} | img_MSE: {avg_il:.6f} | "
                f"train_acc: {avg_train_acc * 100:.2f}% | PSNR: {epoch_psnr:.1f}dB | D_loss: {avg_dl:.4f}"
            )

    # Save final validated checkpoint
    ckpt_path = DEFAULT_CHECKPOINT_DIR / "gan_steganography.pt"
    checkpoint_data = {
        "version": MODEL_VERSION,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "discriminator": discriminator.state_dict(),
        "secret_bits": args.secret_bits,
        "image_size": args.image_size,
        "hidden_size": args.hidden_size,
        "epoch": args.epochs,
        "val_acc": best_val_acc,
        "psnr": final_psnr,
        "arch": "dense",
    }
    torch.save(checkpoint_data, ckpt_path)
    print("\n" + "=" * 70)
    print(f"Training completed successfully!")
    print(f"Checkpoint saved to: {ckpt_path}")
    print(f"Final Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Final PSNR: {final_psnr:.1f} dB")
    print(f"Checkpoint architecture: Dense GAN (version {MODEL_VERSION})")
    print("=" * 70)


if __name__ == "__main__":
    main()
