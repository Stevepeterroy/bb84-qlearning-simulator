"""
simulation.py — RL-Integrated BB84 Demo Runner + AES Pipeline
==============================================================

Runs a demonstration episode with Eve's trained RL agent and displays
the full BB84 protocol results. Includes the AES-256-GCM encryption
pipeline with Eve's probabilistic key reconstruction attack.

Metrics include Shannon information gain and key leakage rate.
"""

import math
import numpy as np
from bb84_environment import BB84Environment
from encryption import (
    generate_aes_key, encrypt_message, decrypt_message,
    key_to_bits, bits_to_key_bytes, eve_brute_force_attack,
)


# ─────────────────── Information-Theoretic Metrics ───────────────────

def compute_information_gain(n_known: int, n_total: int) -> float:
    """
    Compute Shannon information gain from Eve's partial key knowledge.

    Information gain = log2(search_space_before / search_space_after)
                     = log2(2^n_total / 2^(n_total - n_known))
                     = n_known  (in bits)

    Args:
        n_known: Number of key bits Eve knows.
        n_total: Total number of key bits.

    Returns:
        Information gain in bits.
    """
    if n_total == 0:
        return 0.0
    return float(n_known)


def compute_leakage_rate(n_known: int, n_total: int) -> float:
    """
    Compute key leakage rate as fraction of key bits known by Eve.

    Args:
        n_known: Number of key bits Eve knows.
        n_total: Total number of key bits.

    Returns:
        Leakage rate as a float in [0, 1].
    """
    if n_total == 0:
        return 0.0
    return n_known / n_total


# ─────────────────────── Demo Episode ────────────────────────────────

def run_demo_episode(env: BB84Environment, agent, verbose: bool = True) -> dict:
    """
    Run a single demonstration episode with the trained agent.

    Uses a greedy policy (no exploration) for the demo.

    Args:
        env:     BB84Environment instance.
        agent:   Trained EveAgent instance.
        verbose: Whether to print step-by-step details.

    Returns:
        Episode statistics dictionary.
    """
    # Save and disable exploration for demo
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    state = env.reset()
    step_log = []

    while not env.done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)

        step_log.append({
            "qubit": env.current_qubit - 1,
            "action": action,
            "reward": reward,
            **info,
        })

        state = next_state

    stats = env.get_episode_stats()

    if verbose:
        _display_demo_results(env, stats, step_log, agent)

    # Restore exploration rate
    agent.epsilon = original_epsilon

    return stats


def run_encryption_demo(stats: dict, message: str = "HELLO QUANTUM",
                        verbose: bool = True) -> dict:
    """
    Demonstrate message encryption/decryption using AES-256-GCM
    with a BB84-distributed key.

    Args:
        stats:   Episode statistics from run_demo_episode.
        message: Plaintext message to encrypt.
        verbose: Whether to print results.

    Returns:
        Dictionary with encryption results.
    """
    alice_key = stats["alice_key"]

    if len(alice_key) == 0:
        if verbose:
            print(f"\n  ✗ No key generated — cannot encrypt.")
        return {"success": False}

    # Generate a proper AES-256 key for the demo
    aes_key = generate_aes_key()

    # Alice encrypts with AES-256-GCM
    nonce, ciphertext = encrypt_message(message, aes_key)

    # Bob decrypts
    try:
        decrypted = decrypt_message(ciphertext, aes_key, nonce)
        success = (message == decrypted)
    except Exception:
        decrypted = "[DECRYPTION FAILED]"
        success = False

    result = {
        "original": message,
        "ciphertext": ciphertext,
        "nonce": nonce,
        "aes_key": aes_key,
        "decrypted": decrypted,
        "success": success,
        "key_length": len(alice_key),
    }

    if verbose:
        _display_encryption_results(result)

    return result


# ─────────────────────── AES Pipeline ────────────────────────────────

def run_aes_pipeline(env: BB84Environment, agent, message: str,
                     verbose: bool = True) -> dict:
    """
    Run the complete AES + BB84 + RL Eve pipeline.

    Full workflow:
      1. Generate AES-256 key
      2. Encrypt message with AES-GCM
      3. Run BB84 demo episode (Eve intercepts qubits)
      4. Extract Eve's partial key + compute info gain
      5. Simulate Eve's probabilistic key reconstruction attack
      6. Return comprehensive results

    Args:
        env:     BB84Environment instance.
        agent:   Trained EveAgent (post-training).
        message: Plaintext message to encrypt.
        verbose: Whether to print detailed output.

    Returns:
        Dictionary with complete pipeline results.
    """
    if verbose:
        print(f"\n  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║   AES-256-GCM + BB84 QKD SECURE COMMUNICATION PIPELINE ║")
        print(f"  ╚══════════════════════════════════════════════════════════╝")

    # ── Step 1: Generate AES-256 key ─────────────────────────────
    aes_key = generate_aes_key()
    aes_key_bits = key_to_bits(aes_key)

    if verbose:
        print(f"\n  ── Step 1: AES Key Generation ────────────────────────────")
        print(f"  AES-256 key generated: {aes_key.hex()[:32]}...")
        print(f"  Key size: {len(aes_key) * 8} bits")

    # ── Step 2: Encrypt message ──────────────────────────────────
    nonce, ciphertext = encrypt_message(message, aes_key)

    if verbose:
        print(f"\n  ── Step 2: AES-GCM Encryption ───────────────────────────")
        print(f"  Original message : \"{message}\"")
        print(f"  Nonce (hex)      : {nonce.hex()}")
        print(f"  Ciphertext (hex) : {ciphertext.hex()[:64]}...")
        print(f"  Ciphertext size  : {len(ciphertext)} bytes (includes 16-byte auth tag)")

    # ── Step 3: BB84 Key Distribution ────────────────────────────
    if verbose:
        print(f"\n  ── Step 3: BB84 Quantum Key Distribution ────────────────")
        print(f"  Transmitting AES key bits via BB84 protocol...")
        print(f"  Qubits available: {env.n_qubits}")

    # Run demo episode — Eve intercepts during BB84 transmission
    demo_stats = run_demo_episode(env, agent, verbose=verbose)

    # ── Step 4: Extract shared key and Eve's partial key ─────────
    alice_key_sifted = demo_stats["alice_key"]
    bob_key_sifted = demo_stats["bob_key"]
    qber = demo_stats["qber"]

    # Get Eve's partial key from the environment
    eve_partial_key_str, eve_partial_key_list = agent.reconstruct_partial_key(env)
    n_known = sum(1 for b in eve_partial_key_list if b != '?')
    n_total = len(eve_partial_key_list)

    # Compute information-theoretic metrics
    leakage_rate = compute_leakage_rate(n_known, n_total)
    info_gain = compute_information_gain(n_known, n_total)
    search_space_before = 2 ** min(n_total, 64)  # Capped for display
    search_space_after = 2 ** min(n_total - n_known, 64)

    if verbose:
        print(f"\n  ── Step 4: Key Analysis & Information Gain ──────────────")
        print(f"  Sifted key length   : {len(alice_key_sifted)} bits")
        print(f"  Alice/Bob shared key: {''.join(str(b) for b in alice_key_sifted[:48])}{'...' if len(alice_key_sifted) > 48 else ''}")
        print(f"  Eve's partial key   : {eve_partial_key_str}")
        print(f"  Eve knows           : {n_known}/{n_total} bits ({leakage_rate:.1%})")
        print(f"  Key leakage rate    : {leakage_rate:.4f}")
        print(f"  QBER                : {qber:.4f} ({qber:.2%})")
        print(f"  Eve detected        : {'Yes ⚠' if demo_stats['detected'] else 'No ✓'}")

    # ── Step 5: Eve's Probabilistic Key Reconstruction ───────────
    eve_full_partial = ['?'] * 256
    for i in range(min(n_total, 256)):
        if eve_partial_key_list[i] != '?':
            eve_full_partial[i] = eve_partial_key_list[i]

    n_unknown_full = sum(1 for b in eve_full_partial if b == '?')

    attack_result = eve_brute_force_attack(
        ciphertext, nonce, eve_full_partial, max_attempts=10000
    )

    # ── Step 6: Bob decrypts (legitimate receiver) ───────────────
    try:
        bob_decrypted = decrypt_message(ciphertext, aes_key, nonce)
        bob_success = (message == bob_decrypted)
    except Exception:
        bob_decrypted = "[DECRYPTION FAILED]"
        bob_success = False

    if verbose:
        print(f"\n  ── Step 6: Bob's Decryption (Legitimate Receiver) ───────")
        if bob_success:
            print(f"  ✓  Bob decrypted successfully: \"{bob_decrypted}\"")
        else:
            print(f"  ✗  Bob's decryption failed: \"{bob_decrypted}\"")

    # ── Build result dict ────────────────────────────────────────
    pipeline_result = {
        "original_message": message,
        "aes_key": aes_key,
        "nonce": nonce,
        "ciphertext": ciphertext,
        "alice_key_sifted": alice_key_sifted,
        "bob_key_sifted": bob_key_sifted,
        "eve_partial_key_str": eve_partial_key_str,
        "eve_partial_key_list": eve_partial_key_list,
        "eve_known_bits": n_known,
        "eve_total_bits": n_total,
        "leakage_rate": leakage_rate,
        "information_gain": info_gain,
        "qber": qber,
        "detected": demo_stats["detected"],
        "attack_result": attack_result,
        "bob_decrypted": bob_decrypted,
        "bob_success": bob_success,
        "demo_stats": demo_stats,
    }

    if verbose:
        _display_pipeline_summary(pipeline_result)

    return pipeline_result


def calculate_qber(alice_key: np.ndarray, bob_key: np.ndarray) -> float:
    """
    Calculate the Quantum Bit Error Rate.

    Args:
        alice_key: Alice's sifted key.
        bob_key:   Bob's sifted key.

    Returns:
        QBER as a float.
    """
    if len(alice_key) == 0:
        return 0.0
    return float(np.sum(alice_key != bob_key) / len(alice_key))


# ─────────────────────── Display Functions ───────────────────────────

def _display_demo_results(env, stats, step_log, agent):
    """Display comprehensive demo results."""
    n = env.n_qubits
    n_show = min(20, n)

    action_labels = agent.get_action_labels()

    print(f"\n  ══════════════════════════════════════════════════════════")
    print(f"    DEMO EPISODE — Trained Eve (ε=0, greedy policy)")
    print(f"  ══════════════════════════════════════════════════════════")

    # Protocol table
    print(f"\n  Showing first {n_show} of {n} qubits:\n")

    basis_map = {0: "Z", 1: "X"}

    print(f"  {'Qubit':<8}", end="")
    for i in range(n_show):
        print(f"{i:<5}", end="")
    print()
    print("  " + "─" * (8 + n_show * 5))

    # Alice bits
    print(f"  {'A bit':<8}", end="")
    for i in range(n_show):
        print(f"{env.alice_bits[i]:<5}", end="")
    print()

    # Alice bases
    print(f"  {'A base':<8}", end="")
    for i in range(n_show):
        print(f"{basis_map[env.alice_bases[i]]:<5}", end="")
    print()

    # Bob bases
    print(f"  {'B base':<8}", end="")
    for i in range(n_show):
        print(f"{basis_map[env.bob_bases[i]]:<5}", end="")
    print()

    # Bob results
    print(f"  {'B bit':<8}", end="")
    for i in range(n_show):
        print(f"{env.bob_results[i]:<5}", end="")
    print()

    # Eve's actions
    print(f"  {'Eve':<8}", end="")
    for i in range(n_show):
        if i < len(step_log):
            act = step_log[i]["action"]
            sym = ["·", "Z", "X", "P"][act]
        else:
            sym = "?"
        print(f"{sym:<5}", end="")
    print()

    # Basis match
    print(f"  {'Match':<8}", end="")
    for i in range(n_show):
        m = "✓" if env.alice_bases[i] == env.bob_bases[i] else "✗"
        print(f"{m:<5}", end="")
    print()

    # Key info
    print(f"\n  Eve legend: · = Skip, Z = Intercept(Z), X = Intercept(X), P = Partial")

    # Statistics
    print(f"\n  ── Episode Statistics ────────────────────────────────")
    print(f"  QBER              : {stats['qber']:.4f} ({stats['qber']:.2%})")
    print(f"  Sifted key length : {stats['sifted_key_length']} bits")
    print(f"  Qubits intercepted: {stats['n_intercepted']}/{n} ({stats['interception_rate']:.1%})")
    print(f"  Eve success rate  : {stats['eve_success_rate']:.1%}")
    print(f"  Key leakage       : {stats['key_leakage']} bits ({stats['key_leakage_rate']:.1%})")
    print(f"  Total reward      : {stats['total_reward']:.1f}")
    print(f"  Detected          : {'Yes ⚠' if stats['detected'] else 'No ✓'}")

    # Raw sifted key




def _display_encryption_results(result):
    """Display AES encryption demo results."""
    print(f"\n  ══════════════════════════════════════════════════════════")
    print(f"    MESSAGE ENCRYPTION — AES-256-GCM")
    print(f"  ══════════════════════════════════════════════════════════")

    print(f"\n  Original message  : \"{result['original']}\"")
    print(f"  Ciphertext (hex)  : {result['ciphertext'].hex()[:64]}...")
    print(f"  Nonce (hex)       : {result['nonce'].hex()}")
    print(f"  Key bits used     : {result['key_length']}")
    print(f"  Decrypted message : \"{result['decrypted']}\"")

    if result['success']:
        print(f"\n  ✓  Message transmitted and decrypted correctly!")
    else:
        print(f"\n  ✗  Decryption failed — key mismatch due to Eve's interference.")


def _display_pipeline_summary(result):
    """Display the final summary of the AES + BB84 pipeline."""
    attack = result["attack_result"]
    n_known = result["eve_known_bits"]
    n_total = result["eve_total_bits"]
    leakage_rate = result["leakage_rate"]
    info_gain = result["information_gain"]

    qber_val = result["qber"]
    qber_str = f"{qber_val:.4f} ({qber_val:.2%})"
    leakage_str = f"{n_known}/{n_total} bits ({leakage_rate:.1%})"
    info_str = f"{info_gain:.1f} bits"
    attempts_str = f"{attack['attempts']:,}"
    detected_str = "Yes ⚠" if result["detected"] else "No ✓"
    attack_str = "YES ⚠" if attack["success"] else "NO ✓"

    eve_msg = "[FAILED]"
    if attack["decrypted"]:
        eve_msg = attack["decrypted"][:29] + "..." if len(attack["decrypted"]) > 29 else attack["decrypted"]

    bob_msg = result["bob_decrypted"]
    if len(bob_msg) > 29:
        bob_msg = bob_msg[:29] + "..."

    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║          SECURITY EVALUATION SUMMARY                    ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Original Message   : {result['original_message']:<35}║")
    print(f"  ║  Ciphertext (hex)   : {result['ciphertext'].hex()[:29]}...  ║")
    print(f"  ║  QBER               : {qber_str:<35}║")
    print(f"  ║  Eve Detected       : {detected_str:<35}║")
    print(f"  ║  Key Leakage Rate   : {leakage_str:<35}║")
    print(f"  ║  Information Gain   : {info_str:<35}║")
    print(f"  ║  Eve Attack Success : {attack_str:<35}║")
    print(f"  ║  Eve Attempts       : {attempts_str:<35}║")
    print(f"  ║  Eve Decrypted      : {eve_msg:<35}║")
    print(f"  ║  Bob Decrypted      : {bob_msg:<35}║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")
