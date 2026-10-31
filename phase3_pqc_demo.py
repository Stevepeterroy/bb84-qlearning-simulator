"""
phase3_pqc_demo.py — RSA vs Kyber Post-Quantum Cryptography Demo
=================================================================

Side-by-side experiment comparing:

  PART 1 — RSA (Classical)
    RSA-2048 encryption → Shor's algorithm simulation → key recovered → decrypted

  PART 2 — Kyber (Post-Quantum)
    Kyber KEM → shared secret → AES key derived → brute-force attack FAILS

Demonstrates why post-quantum cryptography (lattice-based) is necessary.
"""

import hashlib
import os
import time

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

# ── Kyber library detection ──────────────────────────────────────────
# Try multiple libraries in order of preference, fall back to simulation.

KYBER_BACKEND = None

try:
    from pqcrypto.kem.kyber512 import generate_keypair, encrypt, decrypt
    KYBER_BACKEND = "pqcrypto"
except ImportError:
    pass

if KYBER_BACKEND is None:
    try:
        from kyber_py.kyber import Kyber512
        KYBER_BACKEND = "kyber_py"
    except ImportError:
        pass

if KYBER_BACKEND is None:
    KYBER_BACKEND = "simulated"


def _kyber_keygen():
    """Generate a Kyber keypair using the available backend."""
    if KYBER_BACKEND == "pqcrypto":
        public_key, private_key = generate_keypair()
        return public_key, private_key

    elif KYBER_BACKEND == "kyber_py":
        public_key, private_key = Kyber512.keygen()
        return public_key, private_key

    else:
        # Simulated Kyber — educational demonstration
        private_key = os.urandom(32)
        public_key = os.urandom(800)   # Kyber512 public key ≈ 800 bytes
        return public_key, private_key


def _kyber_encapsulate(public_key):
    """Encapsulate a shared secret using the available backend."""
    if KYBER_BACKEND == "pqcrypto":
        ciphertext, shared_secret = encrypt(public_key)
        return ciphertext, shared_secret

    elif KYBER_BACKEND == "kyber_py":
        ciphertext, shared_secret = Kyber512.enc(public_key)
        return ciphertext, shared_secret

    else:
        # Simulated: ciphertext ≈ 768 bytes, shared secret = 32 bytes
        shared_secret = os.urandom(32)
        ciphertext = os.urandom(768)
        return ciphertext, shared_secret


def _kyber_decapsulate(private_key, ciphertext):
    """Decapsulate the shared secret using the available backend."""
    if KYBER_BACKEND == "pqcrypto":
        return decrypt(private_key, ciphertext)

    elif KYBER_BACKEND == "kyber_py":
        return Kyber512.dec(ciphertext, private_key)

    else:
        # Simulated: in real Kyber this recovers the same shared secret
        return os.urandom(32)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN DEMO
# ═══════════════════════════════════════════════════════════════════════

def run_pqc_demo(message: str = "HELLO QUANTUM"):
    """
    Run the RSA vs Kyber side-by-side security demonstration.

    Args:
        message: Plaintext message to encrypt with RSA.

    Returns:
        Dictionary with demo results for both systems.
    """

    print("\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   PHASE 3 — RSA vs KYBER SECURITY TEST                 ║")
    print("  ╚══════════════════════════════════════════════════════════╝")

    results = {}

    # ──────────────────────────────────────────────────────────────
    #  PART 1 — RSA DEMONSTRATION (Classical Cryptography)
    # ──────────────────────────────────────────────────────────────

    print(f"\n  ── CLASSICAL CRYPTOGRAPHY — RSA ─────────────────────────\n")

    # Generate RSA-2048 keypair
    key = RSA.generate(2048)
    public_key = key.publickey()

    # Encrypt the message
    cipher = PKCS1_OAEP.new(public_key)
    ciphertext = cipher.encrypt(message.encode())

    print(f"  Message            : \"{message}\"")
    print(f"  RSA key size       : 2048 bits")
    print(f"  RSA ciphertext     : {ciphertext[:20].hex()}...")
    print(f"  Ciphertext length  : {len(ciphertext)} bytes")

    # Simulate Shor's algorithm attack
    print(f"\n  Simulating Shor's algorithm attack...")
    time.sleep(1)

    print(f"  → Quantum period-finding executed (simulated)")
    print(f"  → RSA modulus factored: n = p × q")
    print(f"  → Private key (d) recovered!")

    # "Recover" the private key and decrypt
    cipher = PKCS1_OAEP.new(key)
    decrypted = cipher.decrypt(ciphertext)

    print(f"\n  ✓  Decrypted message : \"{decrypted.decode()}\"")
    print(f"\n  Conclusion: RSA becomes VULNERABLE with quantum computers")
    print(f"  (Shor's algorithm breaks RSA in polynomial time)")

    results["rsa"] = {
        "key_size": 2048,
        "ciphertext_length": len(ciphertext),
        "decrypted": decrypted.decode(),
        "vulnerable": True,
    }

    # ──────────────────────────────────────────────────────────────
    #  PART 2 — KYBER DEMONSTRATION (Post-Quantum Cryptography)
    # ──────────────────────────────────────────────────────────────

    print(f"\n  ── POST-QUANTUM CRYPTOGRAPHY — KYBER ────────────────────\n")

    backend_label = {
        "pqcrypto": "pqcrypto (C bindings)",
        "kyber_py": "kyber-py (pure Python)",
        "simulated": "simulated (educational)",
    }
    print(f"  Kyber backend      : {backend_label[KYBER_BACKEND]}")

    # Key generation
    pub_key, priv_key = _kyber_keygen()

    # Encapsulation
    ct, shared_secret_enc = _kyber_encapsulate(pub_key)

    # Decapsulation
    shared_secret_dec = _kyber_decapsulate(priv_key, ct)

    print(f"  Kyber ciphertext   : {len(ct)} bytes")
    print(f"  Shared secret      : established ({len(shared_secret_enc)} bytes)")

    # Derive AES key from shared secret
    aes_key = hashlib.sha256(shared_secret_enc).digest()
    print(f"  AES key derived    : SHA-256(shared_secret) → {aes_key[:8].hex()}...")

    # Simulate brute-force attack on the AES key
    print(f"\n  Attacker trying to recover key (brute force)...")

    attempts = 100_000
    success = False
    start = time.time()

    for i in range(attempts):
        guess = os.urandom(32)
        if guess == aes_key:
            success = True
            break

    elapsed = time.time() - start

    print(f"  Attack attempts    : {attempts:,}")
    print(f"  Time elapsed       : {elapsed:.3f}s")

    if success:
        print(f"  Result             : Key recovered (unexpected!)")
    else:
        print(f"  Result             : Attack FAILED ✗")

    print(f"\n  Search space       : ≈ 2^256 possibilities")
    print(f"  At 10^12 guesses/s : ≈ 10^53 years to brute-force")

    print(f"\n  ✓  Kyber remains SECURE against quantum attacks")
    print(f"  (Based on Module-LWE lattice hardness problem)")

    results["kyber"] = {
        "backend": KYBER_BACKEND,
        "ciphertext_length": len(ct),
        "shared_secret_length": len(shared_secret_enc),
        "attack_attempts": attempts,
        "attack_success": success,
        "secure": True,
    }

    # ──────────────────────────────────────────────────────────────
    #  COMPARISON SUMMARY
    # ──────────────────────────────────────────────────────────────

    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║          PHASE 3 — COMPARISON SUMMARY                  ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║                                                        ║")
    print(f"  ║  RSA-2048 (Classical):                                 ║")
    print(f"  ║  • Security basis   : Integer factorization            ║")
    print(f"  ║  • Quantum attack   : Shor's algorithm (polynomial)   ║")
    print(f"  ║  • Status           : ⚠  BROKEN by quantum computer   ║")
    print(f"  ║                                                        ║")
    print(f"  ║  Kyber-512 (Post-Quantum):                             ║")
    print(f"  ║  • Security basis   : Module-LWE (lattice)            ║")
    print(f"  ║  • Quantum attack   : No known efficient algorithm    ║")
    print(f"  ║  • Status           : ✓  SECURE against quantum       ║")
    print(f"  ║                                                        ║")
    print(f"  ║  Conclusion:                                           ║")
    print(f"  ║  Post-quantum cryptography (Kyber) provides security   ║")
    print(f"  ║  that classical RSA cannot offer in the quantum era.   ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")

    return results


# ── Standalone execution ─────────────────────────────────────────────

if __name__ == "__main__":
    run_pqc_demo("HELLO QUANTUM")
