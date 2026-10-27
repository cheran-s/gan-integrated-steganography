import io
import math
from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from gan_model import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SECRET_BITS,
    MODEL_VERSION,
    DenseDecoder,
    DenseEncoder,
    pack_message,
    tile_bits,
    unpack_message,
)

# Robust paths independent of the terminal current working directory
BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "checkpoints" / "gan_steganography.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# Metrics (PSNR and SSIM)
# ----------------------------------------------------------------------
def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in dB between two images."""
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for PSNR calculation: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM) between two images."""
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch for SSIM calculation: {img1.shape} vs {img2.shape}")

    if img1.ndim == 3:
        ssims = [calculate_ssim(img1[..., i], img2[..., i]) for i in range(img1.shape[2])]
        return float(np.mean(ssims))

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(numerator / max(denominator, 1e-10))


# ----------------------------------------------------------------------
# LSB Baseline Functions
# ----------------------------------------------------------------------
def lsb_encode(arr: np.ndarray, msg: str) -> np.ndarray:
    """Embeds UTF-8 message into image least significant bits with a 4-byte length prefix."""
    import struct

    payload = msg.encode("utf-8")
    packet = struct.pack(">I", len(payload)) + payload
    bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
    flat = arr.reshape(-1).copy()
    if len(bits) > len(flat):
        raise ValueError("Message is too long for this image.")
    flat[: len(bits)] = (flat[: len(bits)] & 0xFE) | bits
    return flat.reshape(arr.shape)


def lsb_decode(arr: np.ndarray) -> str:
    """Extracts UTF-8 message from image least significant bits."""
    import struct

    flat = arr.reshape(-1)
    if len(flat) < 32:
        raise ValueError("Image is too small to contain an LSB message.")
    header_bits = (flat[:32] & 1).astype(np.uint8)
    length = struct.unpack(">I", np.packbits(header_bits).tobytes())[0]

    max_possible_bytes = (len(flat) - 32) // 8
    if length > max_possible_bytes:
        raise ValueError("No valid LSB message found in image.")

    data_bits = (flat[32 : 32 + length * 8] & 1).astype(np.uint8)
    data_bytes = np.packbits(data_bits).tobytes()
    try:
        return data_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Decoded LSB bytes are not valid UTF-8 text.")


# ----------------------------------------------------------------------
# Checkpoint Loader and Validation
# ----------------------------------------------------------------------
@st.cache_resource
def load_gan_checkpoint(path_str: str, mtime: float):
    """
    Loads and validates the trained GAN checkpoint with architectural compatibility checks.
    """
    ckpt = torch.load(path_str, map_location=DEVICE, weights_only=False)

    version = ckpt.get("version", "1.0")
    if version != MODEL_VERSION or ckpt.get("arch") != "dense":
        raise ValueError(
            f"Incompatible checkpoint version '{version}' found (expected v{MODEL_VERSION} Dense GAN). "
            "Please train the updated model by running 'python train_gan.py'."
        )

    secret_bits = int(ckpt.get("secret_bits", DEFAULT_SECRET_BITS))
    hidden_size = int(ckpt.get("hidden_size", DEFAULT_HIDDEN_SIZE))
    image_size = int(ckpt.get("image_size", DEFAULT_IMAGE_SIZE))

    encoder = DenseEncoder(data_depth=1, hidden_size=hidden_size).to(DEVICE)
    decoder = DenseDecoder(data_depth=1, hidden_size=hidden_size).to(DEVICE)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    metadata = {
        "version": version,
        "arch": ckpt.get("arch", "dense"),
        "secret_bits": secret_bits,
        "image_size": image_size,
        "hidden_size": hidden_size,
        "val_acc": ckpt.get("val_acc", None),
        "psnr": ckpt.get("psnr", None),
        "epoch": ckpt.get("epoch", None),
    }
    return encoder, decoder, metadata


# ----------------------------------------------------------------------
# GAN Embedding and Extraction (Native Resolution Preserving)
# ----------------------------------------------------------------------
def gan_hide_message(
    cover_image: Image.Image,
    message: str,
    encoder: nn.Module,
    secret_bits: int,
    image_size: int,
) -> tuple[Image.Image, Image.Image]:
    """
    Embeds a secret text message into the cover image using the trained GAN encoder.
    For images >= image_size, uses canonical center patch embedding to preserve native resolution.
    For smaller images, resizes to canonical resolution with Lanczos filter.
    """
    bits_np = pack_message(message, secret_bits)
    bits_tensor = torch.from_numpy(bits_np).unsqueeze(0).to(DEVICE)
    msg_map = tile_bits(bits_tensor, image_size, image_size, secret_bits)

    cover_rgb = cover_image.convert("RGB")
    w, h = cover_rgb.size

    if w >= image_size and h >= image_size:
        x0 = (w - image_size) // 2
        y0 = (h - image_size) // 2
        patch = cover_rgb.crop((x0, y0, x0 + image_size, y0 + image_size))

        arr_patch = np.array(patch, dtype=np.uint8, copy=True)
        cover_tensor = torch.from_numpy(arr_patch).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0

        with torch.no_grad():
            stego_patch_tensor = encoder(cover_tensor, msg_map)

        stego_patch_np = (
            stego_patch_tensor.cpu()[0].permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
        ).round().astype(np.uint8)
        stego_patch = Image.fromarray(stego_patch_np)

        stego_image = cover_rgb.copy()
        stego_image.paste(stego_patch, (x0, y0))
        return stego_image, cover_rgb
    else:
        cover_canonical = cover_rgb.resize((image_size, image_size), Image.Resampling.LANCZOS)
        arr = np.array(cover_canonical, dtype=np.uint8, copy=True)
        cover_tensor = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0

        with torch.no_grad():
            stego_tensor = encoder(cover_tensor, msg_map)

        stego_np = (
            stego_tensor.cpu()[0].permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
        ).round().astype(np.uint8)
        stego_image = Image.fromarray(stego_np)
        return stego_image, cover_canonical


def gan_extract_message(
    stego_image: Image.Image,
    decoder: nn.Module,
    secret_bits: int,
    image_size: int,
) -> str:
    """
    Extracts the secret text message from the stego image using the trained GAN decoder.
    If image is >= image_size, extracts from canonical center patch.
    """
    stego_rgb = stego_image.convert("RGB")
    w, h = stego_rgb.size

    if w >= image_size and h >= image_size:
        x0 = (w - image_size) // 2
        y0 = (h - image_size) // 2
        carrier_patch = stego_rgb.crop((x0, y0, x0 + image_size, y0 + image_size))
    else:
        carrier_patch = stego_rgb.resize((image_size, image_size), Image.Resampling.LANCZOS)

    arr = np.array(carrier_patch, dtype=np.uint8, copy=True)
    stego_tensor = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0

    with torch.no_grad():
        logits_map = decoder(stego_tensor)

    return unpack_message(logits_map, secret_bits)


# ----------------------------------------------------------------------
# Streamlit Application User Interface (Simple & Clean)
# ----------------------------------------------------------------------
def run_ui():
    st.set_page_config(page_title="GAN Image Steganography", layout="wide")
    st.title("GAN-Based Image Steganography")
    st.caption("Secure Message Embedding and Extraction Using Deep Learning")

    # Load checkpoint
    encoder = None
    decoder = None
    ckpt_metadata = {}

    if CKPT_PATH.exists():
        try:
            encoder, decoder, ckpt_metadata = load_gan_checkpoint(str(CKPT_PATH), CKPT_PATH.stat().st_mtime)
            st.sidebar.success("Trained GAN ready")
        except Exception as e:
            st.sidebar.error(f"Checkpoint error: {e}")
    else:
        st.sidebar.warning("No checkpoint found. Train via 'python train_gan.py'.")

    # Tabs layout: Hide Message and Extract Message
    tab_hide, tab_extract = st.tabs(["Hide Message", "Extract Message"])

    # ------------------------------------------------------------------
    # Tab 1: Hide Message
    # ------------------------------------------------------------------
    with tab_hide:
        up = st.file_uploader("Upload Cover Image", type=["png", "jpg", "jpeg", "webp"], key="hide_uploader")
        msg = st.text_area("Enter Secret Message", placeholder="Type your secret message here...", height=100)

        if up:
            im = Image.open(up).convert("RGB")
            st.image(im, caption="Cover Image", use_container_width=True)

            if st.button("Hide Message", type="primary"):
                try:
                    if not msg:
                        raise ValueError("Please enter a secret message first.")

                    if encoder is None:
                        raise ValueError("No trained GAN checkpoint found. Run train_gan.py first.")

                    secret_bits = ckpt_metadata.get("secret_bits", DEFAULT_SECRET_BITS)
                    image_size = ckpt_metadata.get("image_size", DEFAULT_IMAGE_SIZE)
                    max_gan_bytes = (secret_bits // 8) - 4
                    msg_bytes = len(msg.encode("utf-8"))
                    if msg_bytes > max_gan_bytes:
                        raise ValueError(f"Message too long: supports up to {max_gan_bytes} UTF-8 bytes.")

                    stego, shown_cover = gan_hide_message(
                        im,
                        msg,
                        encoder,
                        secret_bits,
                        image_size,
                    )

                    # Display comparison side by side
                    col1, col2 = st.columns(2)
                    col1.image(shown_cover, caption="Original Cover Image", use_container_width=True)
                    col2.image(stego, caption="Generated Stego Image", use_container_width=True)

                    # Quality metrics
                    score_psnr = calculate_psnr(np.asarray(shown_cover), np.asarray(stego))
                    score_ssim = calculate_ssim(np.asarray(shown_cover), np.asarray(stego))
                    m1, m2 = st.columns(2)
                    m1.metric("PSNR (dB)", "Infinite" if math.isinf(score_psnr) else f"{score_psnr:.2f}")
                    m2.metric("SSIM", f"{score_ssim:.4f}")

                    # Lossless PNG download
                    buf = io.BytesIO()
                    stego.save(buf, format="PNG")
                    st.download_button(
                        label="Download Stego Image",
                        data=buf.getvalue(),
                        file_name="stego_image.png",
                        mime="image/png",
                    )
                    st.success("Stego image generated successfully.")

                except Exception as e:
                    st.error(str(e))

    # ------------------------------------------------------------------
    # Tab 2: Extract Message
    # ------------------------------------------------------------------
    with tab_extract:
        up2 = st.file_uploader("Upload Stego Image", type=["png", "jpg", "jpeg", "webp"], key="extract_uploader")
        if up2:
            stego_preview = Image.open(up2).convert("RGB")
            st.image(stego_preview, caption="Uploaded Stego Image", use_container_width=True)

            if st.button("Extract Message", type="primary"):
                try:
                    if decoder is None:
                        raise ValueError("No trained GAN checkpoint found.")

                    secret_bits = ckpt_metadata.get("secret_bits", DEFAULT_SECRET_BITS)
                    image_size = ckpt_metadata.get("image_size", DEFAULT_IMAGE_SIZE)
                    result = gan_extract_message(
                        stego_preview,
                        decoder,
                        secret_bits,
                        image_size,
                    )

                    st.success("Message extracted successfully.")
                    st.text_area("Recovered Message", value=result, height=100)

                except Exception as e:
                    st.error(f"Extraction failed: {e}")


if __name__ == "__main__":
    run_ui()
