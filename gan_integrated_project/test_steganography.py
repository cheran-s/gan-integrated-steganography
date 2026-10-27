import io
import math
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

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
from app import (
    calculate_psnr,
    calculate_ssim,
    gan_extract_message,
    gan_hide_message,
    load_gan_checkpoint,
    lsb_decode,
    lsb_encode,
)


def run_all_tests():
    print("=" * 75)
    print("GAN-Based Image Steganography - Comprehensive Automated Test Suite")
    print("=" * 75)
    passed_tests = 0
    total_tests = 0

    def record_result(test_name: str, passed: bool, details: str = ""):
        nonlocal passed_tests, total_tests
        total_tests += 1
        status = "PASS" if passed else "FAIL"
        if passed:
            passed_tests += 1
            print(f"[{status}] {test_name} {('- ' + details) if details else ''}")
        else:
            print(f"[{status}] {test_name} - FAILED! {details}")

    # -------------------------------------------------------------
    # Test 1: Model Import and Instantiation
    # -------------------------------------------------------------
    try:
        enc = DenseEncoder(data_depth=1, hidden_size=DEFAULT_HIDDEN_SIZE)
        dec = DenseDecoder(data_depth=1, hidden_size=DEFAULT_HIDDEN_SIZE)
        disc = Discriminator(hidden_size=32)

        x = torch.rand(2, 3, 128, 128)
        msg_map = torch.randint(0, 2, (2, 1, 128, 128)).float()
        stego = enc(x, msg_map)
        logits = dec(stego)
        d_out = disc(stego)

        shapes_ok = (
            stego.shape == (2, 3, 128, 128)
            and logits.shape == (2, 1, 128, 128)
            and d_out.shape == (2, 1)
        )
        record_result(
            "Test 1: Model Import & Architecture Forward Pass",
            shapes_ok,
            f"Encoder {stego.shape}, Decoder {logits.shape}, Discriminator {d_out.shape}",
        )
    except Exception as e:
        record_result("Test 1: Model Import & Architecture Forward Pass", False, str(e))

    # -------------------------------------------------------------
    # Test 2: Message Packing & Unpacking Helper Functions
    # -------------------------------------------------------------
    try:
        test_str = "Test Payload 2026"
        packed = pack_message(test_str, DEFAULT_SECRET_BITS)
        is_256_bits = len(packed) == DEFAULT_SECRET_BITS and set(np.unique(packed)).issubset({0.0, 1.0})

        # Simulate ideal decoder output with tile_bits
        bits_t = torch.from_numpy(packed).unsqueeze(0)
        simulated_logits = tile_bits(bits_t, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, DEFAULT_SECRET_BITS) * 10.0 - 5.0
        unpacked = unpack_message(simulated_logits, DEFAULT_SECRET_BITS)
        matches = unpacked == test_str
        record_result(
            "Test 2: Payload Packing and Spatial Extraction Logic",
            is_256_bits and matches,
            f"Packed 256 bits, extracted '{unpacked}'",
        )
    except Exception as e:
        record_result("Test 2: Payload Packing and Spatial Extraction Logic", False, str(e))

    # -------------------------------------------------------------
    # Test 3: Empty Message Handling
    # -------------------------------------------------------------
    try:
        empty_str = ""
        packed_empty = pack_message(empty_str, DEFAULT_SECRET_BITS)
        bits_empty = torch.from_numpy(packed_empty).unsqueeze(0)
        sim_empty = tile_bits(bits_empty, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, DEFAULT_SECRET_BITS) * 10.0 - 5.0
        unpacked_empty = unpack_message(sim_empty, DEFAULT_SECRET_BITS)
        record_result(
            "Test 3: Empty Message Handling",
            unpacked_empty == "",
            f"Decoded empty string correctly: {unpacked_empty!r}",
        )
    except Exception as e:
        record_result("Test 3: Empty Message Handling", False, str(e))

    # -------------------------------------------------------------
    # Test 4: Maximum Length Message Handling (28 bytes)
    # -------------------------------------------------------------
    try:
        max_str = "A" * 28  # Exactly 28 UTF-8 bytes
        packed_max = pack_message(max_str, DEFAULT_SECRET_BITS)
        bits_max = torch.from_numpy(packed_max).unsqueeze(0)
        sim_max = tile_bits(bits_max, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, DEFAULT_SECRET_BITS) * 10.0 - 5.0
        unpacked_max = unpack_message(sim_max, DEFAULT_SECRET_BITS)
        record_result(
            "Test 4: Maximum Length Payload Handling (28 bytes)",
            unpacked_max == max_str,
            f"Length: {len(unpacked_max)} bytes",
        )
    except Exception as e:
        record_result("Test 4: Maximum Length Payload Handling", False, str(e))

    # -------------------------------------------------------------
    # Test 5: Oversized Message Rejection (> 28 bytes)
    # -------------------------------------------------------------
    try:
        oversized_str = "This message is definitely longer than twenty-eight bytes!"
        threw_error = False
        try:
            pack_message(oversized_str, DEFAULT_SECRET_BITS)
        except ValueError as ve:
            threw_error = True
            err_msg = str(ve)
        record_result(
            "Test 5: Oversized Message Rejection (> 28 bytes)",
            threw_error,
            f"Properly raised ValueError: '{err_msg[:45]}...'",
        )
    except Exception as e:
        record_result("Test 5: Oversized Message Rejection", False, str(e))

    # -------------------------------------------------------------
    # Test 6: Unicode & Emoji Handling
    # -------------------------------------------------------------
    try:
        unicode_str = "Café ☕ 🔐"
        packed_u = pack_message(unicode_str, DEFAULT_SECRET_BITS)
        bits_u = torch.from_numpy(packed_u).unsqueeze(0)
        sim_u = tile_bits(bits_u, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, DEFAULT_SECRET_BITS) * 10.0 - 5.0
        unpacked_u = unpack_message(sim_u, DEFAULT_SECRET_BITS)
        record_result(
            "Test 6: Unicode & Emoji Character Handling",
            unpacked_u == unicode_str,
            f"Input: {repr(unicode_str)} -> Extracted: {repr(unpacked_u)}",
        )
    except Exception as e:
        record_result("Test 6: Unicode & Emoji Character Handling", False, str(e))

    # -------------------------------------------------------------
    # Test 7: Image Quality Metrics (PSNR and SSIM)
    # -------------------------------------------------------------
    try:
        arr1 = np.full((128, 128, 3), 120, dtype=np.uint8)
        arr2 = arr1.copy()
        psnr_inf = calculate_psnr(arr1, arr2)
        ssim_one = calculate_ssim(arr1, arr2)

        arr3 = np.clip(arr1.astype(int) + 2, 0, 255).astype(np.uint8)
        psnr_val = calculate_psnr(arr1, arr3)
        ssim_val = calculate_ssim(arr1, arr3)

        metrics_valid = (
            math.isinf(psnr_inf)
            and abs(ssim_one - 1.0) < 1e-4
            and 35.0 < psnr_val < 50.0
            and 0.95 < ssim_val <= 1.0
        )
        record_result(
            "Test 7: PSNR & SSIM Image Quality Metrics Calculation",
            metrics_valid,
            f"Identical PSNR={psnr_inf}, SSIM={ssim_one:.4f} | Perturbed PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.4f}",
        )
    except Exception as e:
        record_result("Test 7: PSNR & SSIM Metrics Calculation", False, str(e))

    # -------------------------------------------------------------
    # Test 8: LSB Baseline Encode & Decode Round-trip
    # -------------------------------------------------------------
    try:
        lsb_cover = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        test_lsb_msg = "Hello LSB Baseline!"
        stego_lsb = lsb_encode(lsb_cover, test_lsb_msg)
        recovered_lsb = lsb_decode(stego_lsb)
        record_result(
            "Test 8: LSB Baseline Encode & Decode Round-trip",
            recovered_lsb == test_lsb_msg,
            f"Extracted: '{recovered_lsb}'",
        )
    except Exception as e:
        record_result("Test 8: LSB Baseline Round-trip", False, str(e))

    # -------------------------------------------------------------
    # Test 9: LSB Baseline Oversized Message Rejection
    # -------------------------------------------------------------
    try:
        tiny_img = np.zeros((4, 4, 3), dtype=np.uint8)  # 48 bytes = 48 bits total
        threw_lsb_err = False
        try:
            lsb_encode(tiny_img, "This text exceeds tiny 4x4 image capacity")
        except ValueError:
            threw_lsb_err = True
        record_result(
            "Test 9: LSB Baseline Oversized Message Rejection",
            threw_lsb_err,
            "Properly rejected message exceeding image bit capacity",
        )
    except Exception as e:
        record_result("Test 9: LSB Baseline Oversized Message Rejection", False, str(e))

    # -------------------------------------------------------------
    # Test 10: Incompatible / Legacy Checkpoint Detection
    # -------------------------------------------------------------
    try:
        # Create a mock legacy checkpoint (v1.0)
        legacy_data = {"version": "1.0", "secret_bits": 256, "image_size": 64, "arch": "old_cnn"}
        mock_path = BASE_DIR / "checkpoints" / "mock_legacy.pt"
        mock_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(legacy_data, mock_path)

        rejected = False
        try:
            load_gan_checkpoint(str(mock_path), mock_path.stat().st_mtime)
        except ValueError as ve:
            if "Incompatible checkpoint" in str(ve):
                rejected = True
        if mock_path.exists():
            mock_path.unlink()

        record_result(
            "Test 10: Incompatible Checkpoint Rejection & Safety Guard",
            rejected,
            "Detected outdated/incompatible checkpoint and prevented silent failures",
        )
    except Exception as e:
        record_result("Test 10: Incompatible Checkpoint Rejection", False, str(e))

    # -------------------------------------------------------------
    # Test 11: CPU Training Smoke Test & Checkpoint Save/Load
    # -------------------------------------------------------------
    try:
        print("\nRunning CPU mini training smoke test (5 steps)...")
        enc = DenseEncoder(1, 16)
        dec = DenseDecoder(1, 16)
        disc = Discriminator(16)
        opt_g = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
        opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)

        cover = torch.rand(2, 3, 64, 64)
        for _ in range(5):
            bits = torch.randint(0, 2, (2, 256)).float()
            msg_map = tile_bits(bits, 64, 64, 256)

            with torch.no_grad():
                fake = enc(cover, msg_map)
            d_l = F.binary_cross_entropy_with_logits(disc(cover), torch.ones_like(disc(cover))) + \
                  F.binary_cross_entropy_with_logits(disc(fake), torch.zeros_like(disc(fake)))
            opt_d.zero_grad()
            d_l.backward()
            opt_d.step()

            fake = enc(cover, msg_map)
            pred = dec(fake)
            g_l = F.binary_cross_entropy_with_logits(pred, msg_map) + 10.0 * F.mse_loss(fake, cover)
            opt_g.zero_grad()
            g_l.backward()
            opt_g.step()

        # Save and reload smoke checkpoint
        smoke_path = BASE_DIR / "checkpoints" / "smoke_test.pt"
        torch.save(
            {
                "version": MODEL_VERSION,
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "discriminator": disc.state_dict(),
                "secret_bits": 256,
                "image_size": 64,
                "hidden_size": 16,
                "arch": "dense",
            },
            smoke_path,
        )
        loaded_enc, loaded_dec, meta = load_gan_checkpoint(str(smoke_path), smoke_path.stat().st_mtime)
        if smoke_path.exists():
            smoke_path.unlink()

        record_result(
            "Test 11: CPU Training Smoke Test & Checkpoint Serialization",
            loaded_enc is not None and meta["hidden_size"] == 16,
            "Completed 5 optimization steps, serialized, and successfully reloaded",
        )
    except Exception as e:
        record_result("Test 11: CPU Training Smoke Test", False, str(e))

    # -------------------------------------------------------------
    # Test 12: Trained GAN Round-trip on Actual Checkpoint (if exists)
    # -------------------------------------------------------------
    main_ckpt = BASE_DIR / "checkpoints" / "gan_steganography.pt"
    if main_ckpt.exists():
        try:
            enc, dec, meta = load_gan_checkpoint(str(main_ckpt), main_ckpt.stat().st_mtime)
            test_cover = Image.new("RGB", (128, 128), color=(130, 160, 200))
            test_phrase = "SecretGAN 2026!"

            stego_img, _ = gan_hide_message(test_cover, test_phrase, enc, meta["secret_bits"], meta["image_size"])

            # Test lossless PNG save and re-load (quantization validation)
            buf = io.BytesIO()
            stego_img.save(buf, format="PNG")
            buf.seek(0)
            reloaded_png = Image.open(buf).convert("RGB")

            recovered_phrase = gan_extract_message(reloaded_png, dec, meta["secret_bits"], meta["image_size"])
            psnr_score = calculate_psnr(np.asarray(test_cover), np.asarray(reloaded_png))

            roundtrip_ok = recovered_phrase == test_phrase
            record_result(
                "Test 12: End-to-End Trained GAN PNG Round-trip on Disk Checkpoint",
                roundtrip_ok,
                f"Input: {test_phrase!r} -> Recovered: {recovered_phrase!r} | Stego PSNR: {psnr_score:.1f} dB",
            )
        except Exception as e:
            record_result("Test 12: End-to-End Trained GAN PNG Round-trip", False, str(e))
    else:
        print("[SKIP] Test 12: End-to-End Trained GAN (checkpoints/gan_steganography.pt not yet trained)")

    # -------------------------------------------------------------
    # Test 13: Absolute Path Independence across Modules
    # -------------------------------------------------------------
    try:
        from gan_model import BASE_DIR as G_BASE
        from train_gan import BASE_DIR as T_BASE
        from app import BASE_DIR as A_BASE
        paths_match = (G_BASE == BASE_DIR) and (T_BASE == BASE_DIR) and (A_BASE == BASE_DIR)
        record_result(
            "Test 13: Absolute Path Independence across Modules",
            paths_match,
            f"Resolved uniformly to: {BASE_DIR}",
        )
    except Exception as e:
        record_result("Test 13: Absolute Path Independence", False, str(e))

    # -------------------------------------------------------------
    # Test 14: Native-Resolution High-Fidelity Embedding & Aspect Ratio Preservation
    # -------------------------------------------------------------
    if main_ckpt.exists():
        try:
            enc, dec, meta = load_gan_checkpoint(str(main_ckpt), main_ckpt.stat().st_mtime)
            # Create a non-square, high-resolution test image (435 x 705)
            high_res_cover = Image.new("RGB", (435, 705), color=(140, 175, 215))
            test_msg = "HighRes Fidelity!"

            stego_high_res, comp_cover = gan_hide_message(
                high_res_cover, test_msg, enc, meta["secret_bits"], meta["image_size"]
            )

            # Validate dimensions and aspect ratio preservation
            dims_preserved = (stego_high_res.size == high_res_cover.size == (435, 705))

            # Validate PNG lossless export and reload
            buf = io.BytesIO()
            stego_high_res.save(buf, format="PNG")
            buf.seek(0)
            reloaded_stego = Image.open(buf).convert("RGB")

            # Validate message extraction
            recovered_msg = gan_extract_message(reloaded_stego, dec, meta["secret_bits"], meta["image_size"])
            extraction_ok = recovered_msg == test_msg

            # Validate quality metrics (PSNR >= 35 dB and SSIM >= 0.95)
            psnr_val = calculate_psnr(np.asarray(high_res_cover), np.asarray(reloaded_stego))
            ssim_val = calculate_ssim(np.asarray(high_res_cover), np.asarray(reloaded_stego))
            quality_ok = psnr_val >= 35.0 and ssim_val >= 0.95

            record_result(
                "Test 14: Native-Resolution Embedding & Aspect Ratio Preservation",
                dims_preserved and extraction_ok and quality_ok,
                f"Dims: {stego_high_res.size} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | Recovered: {recovered_msg!r}",
            )
        except Exception as e:
            record_result("Test 14: Native-Resolution Embedding", False, str(e))
    else:
        print("[SKIP] Test 14: Native-Resolution Embedding (checkpoint not found)")

    # -------------------------------------------------------------
    # Test 15: Emoji & Decorative Symbol Verification in User Interface
    # -------------------------------------------------------------
    try:
        app_code = (BASE_DIR / "app.py").read_text(encoding="utf-8")
        # Check for decorative emoji code points
        forbidden_emojis = ["🔐", "🔒", "🔓", "📖", "💾", "🚀", "✨", "🎯", "🤖", "⚡"]
        found_emojis = [emoji for emoji in forbidden_emojis if emoji in app_code]
        no_emojis = len(found_emojis) == 0
        record_result(
            "Test 15: Emoji-Free Academic Interface Compliance",
            no_emojis,
            "Clean professional typography verified; 0 decorative emojis detected" if no_emojis else f"Found emojis: {found_emojis}",
        )
    except Exception as e:
        record_result("Test 15: Emoji-Free Academic Interface Compliance", False, str(e))

    # -------------------------------------------------------------
    # Test 16: LSB Native-Resolution & High-Capacity Round-trip
    # -------------------------------------------------------------
    try:
        hr_cover = np.zeros((300, 400, 3), dtype=np.uint8)
        hr_cover[:] = [100, 150, 200]
        test_lsb_msg = "Academic Final Year Project Demonstration - LSB Baseline"
        encoded_arr = lsb_encode(hr_cover, test_lsb_msg)
        decoded_lsb_msg = lsb_decode(encoded_arr)
        lsb_ok = (decoded_lsb_msg == test_lsb_msg) and (encoded_arr.shape == hr_cover.shape)
        record_result(
            "Test 16: LSB Baseline Native Resolution & High Capacity Round-trip",
            lsb_ok,
            f"Preserved shape {encoded_arr.shape} | Decoded: {decoded_lsb_msg!r}",
        )
    except Exception as e:
        record_result("Test 16: LSB Baseline Native Resolution", False, str(e))

    print("=" * 75)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} Tests Passed.")
    print("=" * 75)
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

