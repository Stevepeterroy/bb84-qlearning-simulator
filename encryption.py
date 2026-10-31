"""
encryption.py — AES-256-GCM Encryption with Quantum Key Distribution
=====================================================================

Implements AES-256-GCM encryption/decryption for the BB84 QKD pipeline:
  - AES key generation (32-byte / 256-bit)
  - Key ↔ binary conversion for BB84 transmission
  - Encrypt/decrypt using AES-256-GCM (authenticated encryption)
  - Eve's probabilistic key reconstruction + brute-force attack
"""

import os
import itertools
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ─────────────────────── Conversion Helpers ──────────────────────────

def text_to_binary(text: str) -> list[int]:
    """
    Convert a UTF-8 string to a list of bits.

    Each character is encoded as 8 bits (one byte).

    Args:
        text: The plaintext string to convert.

    Returns:
        List of 0s and 1s representing the binary encoding.
    """
    binary = []
    for byte in text.encode("utf-8"):
        for i in range(7, -1, -1):
            binary.append((byte >> i) & 1)
    return binary


def binary_to_text(bits: list[int]) -> str:
    """
    Convert a list of bits back to a UTF-8 string.

    Args:
        bits: List of 0s and 1s (length should be a multiple of 8).

    Returns:
        The decoded string.
    """
    chars = []
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i + 8]:
            byte = (byte << 1) | bit
        chars.append(byte)
    return bytes(chars).decode("utf-8", errors="replace")


# ─────────────────── AES Key ↔ Binary Conversion ────────────────────

def key_to_bits(key_bytes: bytes) -> list[int]:
    """
    Convert AES key bytes to a list of bits (MSB first).

    Args:
        key_bytes: The AES key as bytes (typically 32 bytes for AES-256).

    Returns:
        List of 0s and 1s (256 bits for a 32-byte key).
    """
    bits = []
    for byte in key_bytes:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_key_bytes(bits: list[int]) -> bytes:
    """
    Convert a list of bits back to key bytes.

    Args:
        bits: List of 0s and 1s (length should be a multiple of 8).

    Returns:
        The key as bytes.
    """
    key_bytes = []
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i + 8]:
            byte = (byte << 1) | bit
        key_bytes.append(byte)
    return bytes(key_bytes)


# ─────────────────── AES-256-GCM Encryption / Decryption ────────────

def generate_aes_key() -> bytes:
    """
    Generate a random 256-bit (32-byte) AES key.

    Returns:
        32-byte key suitable for AES-256-GCM.
    """
    return AESGCM.generate_key(bit_length=256)


def encrypt_message(plaintext: str, key: bytes) -> tuple[bytes, bytes]:
    """
    Encrypt a plaintext message using AES-256-GCM.

    AES-GCM provides both confidentiality and authentication.
    A random 12-byte nonce is generated for each encryption.

    Args:
        plaintext: The message string to encrypt.
        key:       32-byte AES-256 key.

    Returns:
        Tuple of (nonce, ciphertext_with_tag).
        The ciphertext includes the 16-byte GCM authentication tag appended.
    """
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # 96-bit nonce recommended for GCM
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return nonce, ciphertext


def decrypt_message(ciphertext: bytes, key: bytes, nonce: bytes) -> str:
    """
    Decrypt AES-256-GCM ciphertext.

    Args:
        ciphertext: The encrypted data (includes GCM auth tag).
        key:        32-byte AES-256 key.
        nonce:      12-byte nonce used during encryption.

    Returns:
        The decrypted plaintext string.

    Raises:
        cryptography.exceptions.InvalidTag: If the key is wrong or
        data has been tampered with.
    """
    aesgcm = AESGCM(key)
    plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext_bytes.decode("utf-8")


# ────────── Probabilistic Key Reconstruction ─────────────────────────

def generate_candidate_keys(partial_key_bits: list,
                            top_k: int = 10000) -> list[list[int]]:
    """
    Generate ranked candidate keys from a partial key.

    For unknown bit positions, candidates are generated with a
    probability-weighted approach:
      1. Highest-ranked candidates fill unknowns with the most
         likely bit values (based on uniform 50/50 prior)
      2. Candidates are sorted by Hamming distance from the
         "most probable" key (all unknowns set to 0)

    This simulates a realistic probabilistic reconstruction
    strategy rather than pure random guessing.

    Args:
        partial_key_bits: List of 256 elements; 0/1 for known, '?' for unknown.
        top_k:            Number of candidate keys to generate.

    Returns:
        List of candidate key bit-lists, ranked by probability
        (most likely first).
    """
    unknown_positions = [i for i, b in enumerate(partial_key_bits) if b == '?']
    n_unknown = len(unknown_positions)

    if n_unknown == 0:
        return [list(partial_key_bits)]

    candidates = []
    total_combos = 2 ** n_unknown

    if total_combos <= top_k:
        # Enumerate all combinations, sorted by number of 1-bits
        # (lower Hamming weight = more probable under uniform prior)
        all_combos = list(itertools.product([0, 1], repeat=n_unknown))
        all_combos.sort(key=lambda c: sum(c))

        for combo in all_combos:
            candidate = list(partial_key_bits)
            for pos, bit_val in zip(unknown_positions, combo):
                candidate[pos] = bit_val
            candidates.append(candidate)
    else:
        # Generate top_k candidates using biased sampling
        # Strategy: start with low-Hamming-weight candidates, then random
        seen = set()

        # Phase 1: Systematic low-weight candidates (most probable)
        for n_ones in range(min(n_unknown, 8)):
            if len(candidates) >= top_k // 2:
                break
            # Generate all combinations with exactly n_ones bits set
            for positions in itertools.combinations(range(n_unknown), n_ones):
                if len(candidates) >= top_k // 2:
                    break
                combo = [0] * n_unknown
                for p in positions:
                    combo[p] = 1
                combo_tuple = tuple(combo)
                if combo_tuple not in seen:
                    seen.add(combo_tuple)
                    candidate = list(partial_key_bits)
                    for pos, bit_val in zip(unknown_positions, combo):
                        candidate[pos] = bit_val
                    candidates.append(candidate)

        # Phase 2: Random sampling for remaining slots
        while len(candidates) < top_k:
            combo = tuple(np.random.randint(0, 2, size=n_unknown))
            if combo not in seen:
                seen.add(combo)
                candidate = list(partial_key_bits)
                for pos, bit_val in zip(unknown_positions, combo):
                    candidate[pos] = bit_val
                candidates.append(candidate)

    return candidates


# ─────────────── Eve's Brute-Force Partial Key Attack ────────────────

def eve_brute_force_attack(ciphertext: bytes, nonce: bytes,
                           partial_key_bits: list,
                           max_attempts: int = 10000) -> dict:
    """
    Simulate Eve's probabilistic key reconstruction attack.

    Uses ranked candidate keys (probabilistic reconstruction) rather
    than pure random guessing. Candidates are generated using
    generate_candidate_keys() which prioritizes low-Hamming-weight
    completions, then falls back to random sampling.

    Args:
        ciphertext:       AES-GCM ciphertext (with auth tag appended).
        nonce:            12-byte nonce used during encryption.
        partial_key_bits: List of length 256 where each element is
                          0, 1 (known bit) or '?' (unknown bit).
        max_attempts:     Maximum number of candidate keys to try.

    Returns:
        Dictionary with attack results:
          - success:       Whether Eve recovered the plaintext.
          - attempts:      Number of keys tried.
          - unknown_bits:  Number of unknown bits in the partial key.
          - total_combos:  Total possible combinations (2^unknown_bits).
          - decrypted:     Decrypted message if successful, else None.
          - key_found:     The correct key if found, else None.
    """
    unknown_positions = [i for i, b in enumerate(partial_key_bits) if b == '?']
    n_unknown = len(unknown_positions)
    total_combos = 2 ** min(n_unknown, 256)  # Cap display to avoid overflow

    result = {
        "success": False,
        "attempts": 0,
        "unknown_bits": n_unknown,
        "total_combos": total_combos,
        "decrypted": None,
        "key_found": None,
    }

    # Generate ranked candidate keys
    candidates = generate_candidate_keys(partial_key_bits, top_k=max_attempts)

    # Try each candidate key
    for attempt, candidate_bits in enumerate(candidates):
        candidate_key = bits_to_key_bytes(candidate_bits)
        try:
            aesgcm = AESGCM(candidate_key)
            plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, None)
            # Auth tag verified — correct key found!
            result["success"] = True
            result["attempts"] = attempt + 1
            result["decrypted"] = plaintext_bytes.decode("utf-8", errors="replace")
            result["key_found"] = candidate_key
            return result
        except Exception:
            continue

    result["attempts"] = len(candidates)
    return result
