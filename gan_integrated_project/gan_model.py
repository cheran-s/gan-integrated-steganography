import collections
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
GAN-Based Image Steganography - Core Neural Network Architectures

Features:
1. DenseEncoder: Dense convolutional generator with reflection padding and bounded residual.
2. DenseDecoder: Dense convolutional extractor recovering per-pixel message logits.
3. Discriminator: PatchGAN CNN critic distinguishing cover images from generated stego images.
4. Deterministic spatial permutation & tiling: Eliminates boundary convolution edge artifacts
   and uniformly disperses payload bits across all 16,384 pixels for robust majority voting.
5. Redundant 4-byte self-checking header: Allows error-tolerant length extraction.
"""

BASE_DIR = Path(__file__).resolve().parent
MODEL_VERSION = "2.0"
DEFAULT_SECRET_BITS = 256
DEFAULT_IMAGE_SIZE = 128
DEFAULT_HIDDEN_SIZE = 48
MAX_PAYLOAD_BYTES = (DEFAULT_SECRET_BITS // 8) - 4  # 28 bytes

# Deterministic spatial permutation key across 128x128 grid (16,384 positions)
_RNG = np.random.RandomState(1337)
SPATIAL_PERM = torch.from_numpy(_RNG.permutation(DEFAULT_IMAGE_SIZE * DEFAULT_IMAGE_SIZE))
SPATIAL_INV_PERM = torch.argsort(SPATIAL_PERM)


def pack_message(text: str, count: int = DEFAULT_SECRET_BITS) -> np.ndarray:
    """
    Encode a secret UTF-8 text message into a fixed-length bit array.
    
    Format:
    - 4-byte self-checking header: [L, L, L, 255 - L], where L is payload byte length.
    - Up to (count // 8 - 4) payload bytes.
    - Zero padding up to total (count // 8) bytes.
    
    Returns a float32 array of shape (count,) with values in {0.0, 1.0}.
    """
    max_payload = (count // 8) - 4
    payload = text.encode("utf-8")
    if len(payload) > max_payload:
        raise ValueError(
            f"Message too long: GAN mode supports up to {max_payload} UTF-8 bytes "
            f"(received {len(payload)} bytes). Please enter a shorter message."
        )
    length = len(payload)
    header = bytes([length, length, length, (255 - length) & 0xFF])
    packet = header + payload + bytes(max_payload - length)
    bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8)).astype(np.float32)
    return bits


def tile_bits(bits_tensor: torch.Tensor, height: int = DEFAULT_IMAGE_SIZE, width: int = DEFAULT_IMAGE_SIZE, count: int = DEFAULT_SECRET_BITS) -> torch.Tensor:
    """
    Tiles and spatially permutes secret bits across the (height, width) image grid.
    Interleaving across spatial positions eliminates boundary convolution padding artifacts.
    """
    b, _ = bits_tensor.shape
    total_needed = height * width
    reps = total_needed // count

    tiled = bits_tensor.repeat(1, reps)  # (B, total_needed)
    if total_needed == len(SPATIAL_PERM):
        perm = SPATIAL_PERM.to(bits_tensor.device)
        shuffled = tiled[:, perm]
    else:
        shuffled = tiled

    return shuffled.view(b, 1, height, width)


def unpack_message(logits_tensor: torch.Tensor, count: int = DEFAULT_SECRET_BITS) -> str:
    """
    Extract and decode a secret text message from raw decoder output logits.
    Reverses spatial permutation and performs spatial majority voting across all observations.
    """
    max_payload = (count // 8) - 4
    b = logits_tensor.size(0) if logits_tensor.ndim == 4 else 1
    flat = logits_tensor.view(b, -1)
    total_len = flat.size(1)

    if total_len == len(SPATIAL_INV_PERM):
        inv_perm = SPATIAL_INV_PERM.to(logits_tensor.device)
        unshuffled = flat[:, inv_perm]
    else:
        unshuffled = flat

    num_blocks = total_len // count
    if num_blocks == 0:
        raise ValueError("Image tensor is smaller than minimum required payload block.")

    usable = unshuffled.view(b, num_blocks, count)
    # Average across all spatial repetitions to eliminate random noise
    mean_logits = usable.mean(dim=1)[0].detach().cpu().numpy()
    bits = (mean_logits > 0).astype(np.uint8)

    byte_arr = np.packbits(bits).tobytes()
    # Decode redundant 4-byte header
    c1 = int(byte_arr[0])
    c2 = int(byte_arr[1])
    c3 = int(byte_arr[2])
    c4 = int(255 - byte_arr[3])

    counter = collections.Counter([c1, c2, c3, c4])
    length = counter.most_common(1)[0][0]

    if length > max_payload:
        raise ValueError(
            f"Invalid decoded length: {length} bytes (max supported is {max_payload}). "
            "The image may not contain a GAN message or was modified."
        )

    payload_bytes = byte_arr[4 : 4 + length]
    try:
        return payload_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Recovered bytes are not valid UTF-8: {e}")


class DenseEncoder(nn.Module):
    """
    Dense Convolutional Generator with reflection padding and bounded residual.
    Embeds secret bit map into cover image with multi-scale feature propagation.
    """

    def __init__(self, data_depth: int = 1, hidden_size: int = DEFAULT_HIDDEN_SIZE):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden_size, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size + data_depth, hidden_size, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2 + data_depth, hidden_size, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size * 3 + data_depth, 3, 3, padding=1, padding_mode="reflect"),
        )

    def forward(self, cover: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cover: (B, 3, H, W) normalized to [0, 1]
            message: (B, 1, H, W) binary values {0, 1}
        Returns:
            stego: (B, 3, H, W) in [0, 1]
        """
        x1 = self.conv1(cover)
        x2 = self.conv2(torch.cat([x1, message], dim=1))
        x3 = self.conv3(torch.cat([x1, x2, message], dim=1))
        residual = torch.tanh(self.conv4(torch.cat([x1, x2, x3, message], dim=1))) * 0.08
        return torch.clamp(cover + residual, 0.0, 1.0)


class DenseDecoder(nn.Module):
    """
    Dense Convolutional Extractor with reflection padding.
    Extracts the embedded secret bits from container images.
    """

    def __init__(self, data_depth: int = 1, hidden_size: int = DEFAULT_HIDDEN_SIZE):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden_size, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size * 3, data_depth, 3, padding=1, padding_mode="reflect"),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) container image
        Returns:
            logits: (B, 1, H, W) raw unthresholded logits
        """
        x1 = self.conv1(image)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        return self.conv4(torch.cat([x1, x2, x3], dim=1))


class Discriminator(nn.Module):
    """
    PatchGAN / CNN Critic distinguishing real cover images from generated stego images.
    """

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_size, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size, hidden_size * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 2, hidden_size * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_size * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Aliases for backward compatibility
Encoder = DenseEncoder
Decoder = DenseDecoder
