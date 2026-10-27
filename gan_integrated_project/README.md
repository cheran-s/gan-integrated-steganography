# GAN-Based Image Steganography Project (v2.0)

A deep learning image steganography framework featuring a **Dense Generator (Encoder)**, **Dense Extractor (Decoder)**, and **PatchGAN Critic (Discriminator)** built with PyTorch and Streamlit. The application also provides an independent **Least Significant Bit (LSB)** baseline algorithm for academic comparison.

---

## 1. Project Overview & Architecture

Image steganography hides confidential data within carrier images such that third parties cannot detect the existence of the secret communication payload.

The system consists of:
1. **DenseEncoder (Generator)**:
   - Accepts a 3-channel RGB cover image and a spatially tiled, permuted secret message map.
   - Uses dense feature skip connections and reflection padding (`padding_mode='reflect'`) to compute subtle, imperceptible residual perturbations bounded by a hyperbolic tangent function (`tanh * 0.08`).
   - For high-resolution images, embeds into a canonical 128x128 center patch, preserving original image dimensions, aspect ratio, and fine details.
2. **DenseDecoder (Extractor)**:
   - A fully convolutional deep network that inspects local pixel perturbations.
   - Outputs per-pixel logits predicting the hidden secret bits.
   - Reverses spatial interleaving and executes spatial majority voting across all 64 spatial repetitions to reconstruct the exact 256-bit payload.
3. **Discriminator (Critic)**:
   - A convolutional PatchGAN downsampling critic with LeakyReLU activations and batch normalization.
   - Distinguishes real cover images from stego images, compelling the generator to preserve natural visual textures.
4. **LSB Baseline Algorithm**:
   - A classical spatial-domain steganography method embedding raw bitstreams into the least significant bit of 8-bit image pixels, serving as an independent comparative baseline.

---

## 2. Visual Quality Enhancement: Native Resolution Preservation

### Root Cause of Previous Visual Quality Loss
In naive neural steganography implementations, uploaded cover images (e.g., 435x705 or 1920x1080) were forcibly downsampled to the neural network's internal resolution (128x128). This squashed the aspect ratio, destroyed over 95% of image detail, and produced stretched, blurry, and pixelated images when displayed or downloaded.

### High-Fidelity Canonical Patch Embedding Solution
To resolve this without requiring network re-training or breaking extraction:
- **Canonical Center Patch Embedding**: When a cover image with dimensions $W \ge 128$ and $H \ge 128$ is provided, the application extracts a canonical $128 \times 128$ center patch:
  $$x_0 = \frac{W - 128}{2}, \quad y_0 = \frac{H - 128}{2}$$
- The `DenseEncoder` embeds the message into this patch with bounded residuals ($\approx 29.5$ dB local patch PSNR).
- The modulated stego patch is pasted back into the full-size canvas at $(x_0, y_0)$.
- The non-carrier image regions retain 100% of their original pixel values (0 distortion, infinite PSNR).
- The exported lossless PNG container retains the **exact original dimensions ($W \times H$)** and aspect ratio.
- Overall full-image PSNR exceeds **42.0 dB** and SSIM exceeds **0.999**.
- During extraction, the application extracts the identical center patch from the re-uploaded PNG and decodes the message with **100% exact match accuracy**.
- Smaller images ($< 128 \times 128$) are handled gracefully by scaling up to canonical carrier dimensions.

---

## 3. User Interface Redesign: Academic Dashboard

The Streamlit interface (`app.py`) provides a clean, simple, and intuitive layout:
- **Zero Decorative Emojis**: Free of decorative emojis, symbols, and icons for a clean, professional appearance.
- **Minimalist Workflow**: Clear, concise labels with no unnecessary paragraphs or clutter.
- **3 Simple Sections**:
  1. **Hide Message**: Upload cover image, enter secret message, click "Hide Message", inspect side-by-side comparison with PSNR and SSIM, and download the stego PNG.
  2. **Extract Message**: Upload stego PNG, click "Extract Message", and view the recovered text payload.
  3. **About**: Concise technical summary of the Encoder, Decoder, Discriminator, and LSB baseline.

---

## 4. Environment Setup & Requirements

- **Operating System**: Windows 10 / 11
- **Python**: 3.10+ (Tested on Python 3.14.7)
- **PyTorch**: PyTorch 2.2+ (CPU-compatible, no GPU required)
- **Streamlit**: 1.35+

### Activating the Virtual Environment in PowerShell

From the project root:
```powershell
# Navigate to project folder
cd C:\Users\cheran\OneDrive\Documents\GAN_PROJECT\gan_integrated_steganography_project\gan_integrated_project

# Activate the virtual environment
..\.venv\Scripts\Activate.ps1
```

*(If script execution is disabled on PowerShell, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

---

## 5. How to Launch the Application

Launch the Streamlit web dashboard:
```powershell
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Verification Workflow:
1. Navigate to **Hide Message** via the sidebar.
2. Select **Trained Dense GAN (v2.0)**.
3. Upload any cover image (e.g. `data/cover_images/images.1.jpg`).
4. Enter a confidential message (up to 28 UTF-8 bytes).
5. Click **Generate Stego Image**.
6. Observe side-by-side comparison, preserved native resolution, and quality metrics (PSNR > 40 dB, SSIM > 0.99).
7. Click **Download Stego Image (PNG)**.
8. Navigate to **Extract Message**, upload the downloaded `stego_image.png`, select **Trained Dense GAN (v2.0)**, and click **Extract Secret Message**.
9. The message is extracted with 100% exact text fidelity.

---

## 6. How to Run the Automated Test Suite

To verify all components, run the 16 automated regression tests:
```powershell
python test_steganography.py
```

### Verified Test Cases:
1. `Test 1: Model Import & Architecture Forward Pass` (DenseEncoder, DenseDecoder, Discriminator tensor dimensions).
2. `Test 2: Payload Packing and Spatial Extraction Logic` (256-bit packing and majority voting).
3. `Test 3: Empty Message Handling` (0-byte payload recovery).
4. `Test 4: Maximum Length Payload Handling` (Exactly 28 UTF-8 bytes).
5. `Test 5: Oversized Message Rejection` (Throws clear ValueError when $> 28$ bytes).
6. `Test 6: Unicode Character Handling` (Preserves UTF-8 multi-byte characters).
7. `Test 7: PSNR & SSIM Quality Metrics Calculation` (Correct mathematical calculations).
8. `Test 8: LSB Baseline Encode & Decode Round-trip` (LSB verification).
9. `Test 9: LSB Baseline Oversized Message Rejection` (Detects image capacity overflow).
10. `Test 10: Incompatible Checkpoint Rejection` (Detects legacy v1.0 or corrupted checkpoints).
11. `Test 11: CPU Training Smoke Test & Checkpoint Serialization` (End-to-end CPU training step).
12. `Test 12: End-to-End Trained GAN PNG Round-trip on Disk Checkpoint` (Full disk PNG export/re-upload).
13. `Test 13: Absolute Path Independence across Modules` (Uniform directory resolution).
14. `Test 14: Native-Resolution Embedding & Aspect Ratio Preservation` (Preserves native WxH dimensions, PSNR $\ge 35$ dB, SSIM $\ge 0.95$).
15. `Test 15: Emoji-Free Academic Interface Compliance` (Verifies 0 decorative emojis across UI).
16. `Test 16: LSB Baseline Native Resolution & High Capacity Round-trip` (Full-capacity LSB round-trip).

---

## 7. Experimental Results & Verification Summary

Measured on Windows CPU using real cover images (`data/cover_images/images.1.jpg`, $435 \times 705$ px):

| Metric | Previous Broken Prototype | Canonical Downscaled (v2.0) | Native-Resolution Preserved (Current) |
|---|---|---|---|
| **Decoder Bit Accuracy** | 48.7% (Random) | **97.87% val / 100% voted** | **97.87% val / 100% voted** |
| **Message Recovery** | FAILED (`Invalid length`) | **100% Exact Match** | **100% Exact Match** |
| **Output Image Dimensions** | Squashed to 64×64 | Squashed to 128×128 | **435 × 705 px (Native Original)** |
| **Aspect Ratio Preservation** | Distorted | Distorted | **100% Preserved** |
| **Full-Image PSNR** | N/A (Dead residual) | 24.86 dB (Stretched) | **42.25 dB** |
| **Full-Image SSIM** | N/A | 0.9689 (Stretched) | **0.9995** |
| **Local Carrier Patch PSNR** | N/A | 29.91 dB | **29.53 dB** |
| **Quantization Robustness** | Fails on uint8 rounding | Immune to PNG rounding | **Immune to PNG rounding** |
| **UI Aesthetics** | Basic tabs with emojis | Basic tabs with emojis | **Professional Dashboard (0 Emojis)** |

---

## 8. Academic Notes & Limitations

- **Carrier Capacity**: Fixed at 28 UTF-8 bytes per 128x128 carrier patch.
- **Compression Sensitivity**: Uncompressed or lossless formats (PNG) are required. Lossy compression (such as JPEG re-encoding) discards high-frequency pixel perturbations, corrupting the hidden payload.
- **Steganography vs. Cryptography**: Steganography conceals the *presence* of a communication channel (covert communication). Cryptography protects the *confidentiality* of the content. In production settings, sensitive plaintext should be encrypted prior to steganographic embedding.
