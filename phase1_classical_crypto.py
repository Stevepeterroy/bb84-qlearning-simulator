"""
phase1_classical_crypto.py — Classical Encryption Demonstration
================================================================

Demonstrates classical RSA cryptography that is vulnerable to
quantum algorithms. Serves as the target for simulated quantum
attacks in Phase 1 of the research simulation.

Provides:
  - RSA-2048 key generation, encryption, and decryption
  - Small RSA (for Shor's attack demo) with recoverable primes
"""

import os
import math
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization


# ─────────────────────── RSA Encryption ──────────────────────────────

def generate_rsa_keypair(key_size: int = 2048):
    """
    Generate an RSA key pair.

    Args:
        key_size: RSA key size in bits (default 2048).

    Returns:
        Tuple of (private_key, public_key).
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )
    public_key = private_key.public_key()
    return private_key, public_key


def rsa_encrypt(plaintext: str, public_key) -> bytes:
    """
    Encrypt a message using RSA-OAEP.

    Args:
        plaintext:  Message string to encrypt.
        public_key: RSA public key object.

    Returns:
        Encrypted ciphertext bytes.
    """
    ciphertext = public_key.encrypt(
        plaintext.encode("utf-8"),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return ciphertext


def rsa_decrypt(ciphertext: bytes, private_key) -> str:
    """
    Decrypt RSA-OAEP ciphertext.

    Args:
        ciphertext:  Encrypted bytes.
        private_key: RSA private key object.

    Returns:
        Decrypted plaintext string.
    """
    plaintext = private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return plaintext.decode("utf-8")


# ──────────────── Small RSA for Shor Attack Demo ─────────────────────

def generate_small_rsa(p: int = 61, q: int = 53):
    """
    Generate a small RSA key pair from known primes (for Shor demo).

    Uses small primes so that Shor's algorithm simulation can
    factor the modulus and recover the private key.

    Args:
        p: First prime factor.
        q: Second prime factor.

    Returns:
        Dictionary with RSA parameters: n, e, d, p, q, phi_n.
    """
    n = p * q
    phi_n = (p - 1) * (q - 1)
    e = 65537

    # Ensure e is coprime with phi_n; fall back to smaller e if needed
    if math.gcd(e, phi_n) != 1:
        e = 17
    if math.gcd(e, phi_n) != 1:
        e = 3

    # Compute private exponent d = e^(-1) mod phi_n
    d = pow(e, -1, phi_n)

    return {"n": n, "e": e, "d": d, "p": p, "q": q, "phi_n": phi_n}


def small_rsa_encrypt(plaintext_int: int, e: int, n: int) -> int:
    """Encrypt an integer using textbook RSA: c = m^e mod n."""
    return pow(plaintext_int, e, n)


def small_rsa_decrypt(ciphertext_int: int, d: int, n: int) -> int:
    """Decrypt an integer using textbook RSA: m = c^d mod n."""
    return pow(ciphertext_int, d, n)


def text_to_int(text: str) -> int:
    """Convert a short string to an integer for small RSA."""
    return int.from_bytes(text.encode("utf-8"), byteorder="big")


def int_to_text(num: int) -> str:
    """Convert an integer back to a string."""
    length = max(1, (num.bit_length() + 7) // 8)
    return num.to_bytes(length, byteorder="big").decode("utf-8", errors="replace")




# ────────────────── Phase 1 Demo Runner ──────────────────────────────

def run_classical_crypto_demo(message: str = "HELLO QUANTUM") -> dict:
    """
    Run the classical cryptography demonstration.

    Encrypts the message using both RSA and AES, and prepares
    a small RSA instance for the Shor attack simulation.

    Args:
        message: Plaintext message.

    Returns:
        Dictionary with all crypto artifacts for the attack phase.
    """
    print(f"\n  ── Classical Encryption Demonstration ────────────────────")

    # ── RSA-2048 Demo ────────────────────────────────────────────
    print(f"\n  [RSA-2048 Encryption]")
    priv_key, pub_key = generate_rsa_keypair(2048)
    rsa_ct = rsa_encrypt(message, pub_key)
    rsa_pt = rsa_decrypt(rsa_ct, priv_key)

    key_size = priv_key.key_size
    n_pub = priv_key.public_key().public_numbers()
    print(f"  Plaintext          : \"{message}\"")
    print(f"  RSA key size       : {key_size} bits")
    print(f"  RSA modulus (n)    : {str(n_pub.n)[:40]}...")
    print(f"  RSA ciphertext     : {rsa_ct.hex()[:48]}...")
    print(f"  RSA decrypted      : \"{rsa_pt}\"")
    print(f"  ✓ RSA encryption/decryption successful")



    # ── Small RSA for Shor Demo ──────────────────────────────────
    small_msg = message[:4] if len(message) > 4 else message
    small_rsa = generate_small_rsa(p=61, q=53)

    # Use a numeric message that fits within the modulus n
    # For small RSA demo, we encode a short message as integer < n
    msg_int = text_to_int(message[:2]) if text_to_int(message[:2]) < small_rsa["n"] else 42
    small_msg_display = message[:2] if text_to_int(message[:2]) < small_rsa["n"] else f"[integer: 42]"

    small_ct = small_rsa_encrypt(msg_int, small_rsa["e"], small_rsa["n"])

    print(f"\n  [Small RSA for Shor Attack Demo]")
    print(f"  Primes             : p={small_rsa['p']}, q={small_rsa['q']}")
    print(f"  Modulus n          : {small_rsa['n']}")
    print(f"  Public exponent e  : {small_rsa['e']}")
    print(f"  Message            : {small_msg_display} → integer {msg_int}")
    print(f"  Ciphertext         : {small_ct}")

    return {
        "message": message,
        "rsa_ciphertext": rsa_ct,
        "rsa_key_size": key_size,
        "small_rsa": small_rsa,
        "small_msg": small_msg_display,
        "small_msg_int": msg_int,
        "small_ciphertext": small_ct,
    }
