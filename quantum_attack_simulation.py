"""
quantum_attack_simulation.py — Simulated Quantum Attacks on Classical Crypto
=============================================================================

Demonstrates how quantum algorithms threaten classical cryptography:

1. Shor's Algorithm — Factors RSA modulus → recovers private key → decrypts
2. Grover's Algorithm — Quadratic speedup for symmetric key search

Uses Qiskit for quantum circuit demonstration where possible, with
classical fallback for the actual computation.
"""

import math
import time
import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
#  SHOR'S ALGORITHM SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def _classical_gcd(a: int, b: int) -> int:
    """Compute GCD using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a


def _find_order(a: int, n: int) -> int:
    """
    Find the multiplicative order of a mod n.

    This is the quantum part in real Shor's algorithm — the order-finding
    subroutine uses quantum period-finding (QFT). Here we simulate it
    classically for demonstration.

    Args:
        a: Base integer.
        n: Modulus.

    Returns:
        Order r such that a^r ≡ 1 (mod n), or -1 if not found.
    """
    if _classical_gcd(a, n) != 1:
        return -1

    r = 1
    current = a % n
    while current != 1 and r < n:
        current = (current * a) % n
        r += 1

    return r if current == 1 else -1


def _quantum_order_finding_demo(a: int, n: int) -> dict:
    """
    Demonstrate quantum order-finding circuit concept using Qiskit.

    Creates a small demonstration circuit showing the structure of
    the quantum order-finding subroutine (modular exponentiation +
    inverse QFT). The actual period is computed classically.

    Returns:
        Dictionary with circuit info and found order.
    """
    order = _find_order(a, n)

    circuit_info = {
        "a": a,
        "n": n,
        "order": order,
        "qiskit_available": QISKIT_AVAILABLE,
        "circuit_depth": None,
        "n_qubits": None,
    }

    if QISKIT_AVAILABLE:
        # Build a conceptual quantum period-finding circuit
        n_count = max(4, int(np.ceil(np.log2(n))) * 2)
        n_work = max(2, int(np.ceil(np.log2(n))))
        total_qubits = n_count + n_work

        qc = QuantumCircuit(total_qubits, n_count)

        # Hadamard on counting register
        for q in range(n_count):
            qc.h(q)

        # Initialize work register to |1⟩
        qc.x(n_count)
        qc.barrier()

        # Controlled modular exponentiation (simplified demo)
        for q in range(min(n_count, 4)):
            qc.cx(q, n_count)
        qc.barrier()

        # Inverse QFT on counting register (simplified)
        for q in range(n_count // 2):
            qc.swap(q, n_count - 1 - q)
        for q in range(n_count):
            qc.h(q)
        qc.barrier()

        # Measure counting register
        qc.measure(range(n_count), range(n_count))

        circuit_info["circuit_depth"] = qc.depth()
        circuit_info["n_qubits"] = total_qubits
        circuit_info["circuit_str"] = str(qc.draw(output="text"))

    return circuit_info


def shor_attack(n: int, e: int, ciphertext: int) -> dict:
    """
    Simulate Shor's algorithm attack on RSA.

    Steps:
      1. Choose random a < n
      2. Find order r of a mod n (quantum step — simulated classically)
      3. If r is even, compute gcd(a^(r/2) ± 1, n) to find factors
      4. Recover private key d from factors
      5. Decrypt ciphertext

    Args:
        n: RSA modulus (n = p × q).
        e: RSA public exponent.
        ciphertext: Encrypted integer.

    Returns:
        Dictionary with attack results.
    """
    print(f"\n  ── Shor's Algorithm Attack Simulation ────────────────────")
    print(f"  Target RSA modulus : n = {n}")
    print(f"  Public exponent    : e = {e}")
    print(f"  Ciphertext         : {ciphertext}")

    result = {
        "success": False,
        "n": n,
        "p": None,
        "q": None,
        "d": None,
        "decrypted_int": None,
        "decrypted_text": None,
        "attempts": 0,
        "order_found": None,
    }

    start_time = time.time()

    # Try random bases until factorization succeeds
    for attempt in range(1, 20):
        a = np.random.randint(2, n - 1)
        result["attempts"] = attempt

        # Step 1: Check if a shares a factor with n (lucky case)
        g = _classical_gcd(a, n)
        if 1 < g < n:
            p, q = g, n // g
            print(f"\n  Attempt {attempt}: a={a}")
            print(f" gcd(a, n) = {g} is a factor")
            result["p"], result["q"] = p, q
            break

        # Step 2: Quantum order-finding (simulated)
        print(f"\n  Attempt {attempt}: a={a}")
        qof_result = _quantum_order_finding_demo(a, n)
        r = qof_result["order"]

        if qof_result["qiskit_available"] and attempt == 1:
            print(f"  Quantum circuit: {qof_result['n_qubits']} qubits, "
                  f"depth {qof_result['circuit_depth']}")

        if r == -1 or r % 2 != 0:
            print(f"  → Order r={r} {'(odd)' if r and r > 0 else '(not found)'} — retry")
            continue

        print(f"  → Order r={r} (even ✓)")

        # Step 3: Compute candidate factors
        x = pow(a, r // 2, n)

        if x == n - 1:
            print(f"  → a^(r/2) ≡ -1 (mod n) — trivial, retry")
            continue

        p = _classical_gcd(x + 1, n)
        q = _classical_gcd(x - 1, n)

        if p * q == n and p > 1 and q > 1:
            result["p"], result["q"] = sorted([p, q])
            print(f"  → Factors found: p={result['p']}, q={result['q']}")
            break
        else:
            print(f"  → gcd results: {p}, {q} — not valid factors, retry")

    elapsed = time.time() - start_time

    # Step 4: Recover private key and decrypt
    if result["p"] and result["q"]:
        p, q = result["p"], result["q"]
        phi_n = (p - 1) * (q - 1)

        if math.gcd(e, phi_n) == 1:
            d = pow(e, -1, phi_n)
            result["d"] = d
            result["decrypted_int"] = pow(ciphertext, d, n)
            result["success"] = True

            # Convert back to text
            try:
                dec_int = result["decrypted_int"]
                length = max(1, (dec_int.bit_length() + 7) // 8)
                result["decrypted_text"] = dec_int.to_bytes(
                    length, byteorder="big"
                ).decode("utf-8", errors="replace")
            except Exception:
                result["decrypted_text"] = f"[integer: {result['decrypted_int']}]"

        print(f"\n  ✓  SHOR'S ATTACK SUCCESSFUL!")
        print(f"  Recovered factors  : p={p}, q={q}")
        print(f"  Euler's totient    : φ(n) = {phi_n}")
        print(f"  Recovered key (d)  : {result['d']}")
        print(f"  Decrypted integer  : {result['decrypted_int']}")
        print(f"  Decrypted message  : \"{result['decrypted_text']}\"")
        print(f"  Time elapsed       : {elapsed:.4f}s")
        print(f"  Attempts needed    : {result['attempts']}")
    else:
        print(f"\n  ✗  Factorization failed after {result['attempts']} attempts")

    result["elapsed"] = elapsed
    return result


# ═══════════════════════════════════════════════════════════════════════
#  GROVER'S ALGORITHM SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def _grover_circuit_demo(n_qubits: int = 3) -> dict:
    """
    Build and run a small Grover's algorithm circuit using Qiskit.

    Searches for a marked state in a 2^n_qubits search space.

    Args:
        n_qubits: Number of qubits (search space = 2^n_qubits).

    Returns:
        Dictionary with circuit info and measurement results.
    """
    result = {"qiskit_available": QISKIT_AVAILABLE, "n_qubits": n_qubits}

    if not QISKIT_AVAILABLE:
        return result

    N = 2 ** n_qubits
    target = N - 1  # Mark the last state |111...1⟩
    n_iterations = max(1, int(np.pi / 4 * np.sqrt(N)))

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Initialize superposition
    qc.h(range(n_qubits))

    for _ in range(n_iterations):
        # Oracle: flip sign of target state |111...1⟩
        qc.barrier()
        # Multi-controlled Z = H on last, MCX, H on last
        if n_qubits <= 3:
            qc.h(n_qubits - 1)
            if n_qubits == 2:
                qc.cx(0, 1)
            elif n_qubits == 3:
                qc.ccx(0, 1, 2)
            qc.h(n_qubits - 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)

        # Diffusion operator
        qc.barrier()
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        if n_qubits == 2:
            qc.cx(0, 1)
        elif n_qubits == 3:
            qc.ccx(0, 1, 2)
        else:
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    # Run simulation
    sim = AerSimulator()
    job = sim.run(qc, shots=1024)
    counts = job.result().get_counts()

    # Find the most measured state
    top_state = max(counts, key=counts.get)
    top_count = counts[top_state]

    result["iterations"] = n_iterations
    result["search_space"] = N
    result["target_state"] = format(target, f"0{n_qubits}b")
    result["top_measured"] = top_state
    result["top_count"] = top_count
    result["total_shots"] = 1024
    result["success_prob"] = top_count / 1024
    result["circuit_depth"] = qc.depth()

    return result


def grover_search_simulation() -> dict:
    """
    Simulate Grover's algorithm speedup for symmetric key search.

    Compares classical brute-force complexity vs Grover's quantum
    speedup across multiple key sizes (64, 128, 192, 256 bits).

    Returns:
        Dictionary with complexity comparisons and Qiskit demo results.
    """
    key_sizes = [64, 128, 192, 256]
    comparisons = []

    for bits in key_sizes:
        classical = 2 ** bits
        grover = 2 ** (bits // 2)
        speedup = classical / grover

        comparisons.append({
            "key_size": bits,
            "classical_exponent": bits,
            "grover_exponent": bits // 2,
            "classical_ops": classical,
            "grover_ops": grover,
        })

    # Run Qiskit Grover demo on small instance
    demo = _grover_circuit_demo(n_qubits=3)

    return {
        "comparisons": comparisons,
        "key_sizes": key_sizes,
        "grover_demo": demo,
    }


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 1 RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_phase1(message: str = "HELLO QUANTUM") -> dict:
    """
    Run the complete Phase 1: Quantum Attack on Classical Cryptography.

    Steps:
      1. Demonstrate RSA and AES encryption
      2. Simulate Shor's algorithm attack on small RSA
      3. Simulate Grover's search speedup analysis
      4. Print security conclusions

    Args:
        message: Plaintext message to use in demos.

    Returns:
        Dictionary with all Phase 1 results.
    """
    from phase1_classical_crypto import run_classical_crypto_demo

    # Step 1: Classical encryption demo
    crypto_result = run_classical_crypto_demo(message)

    # Step 2: Shor's attack on small RSA
    shor_result = shor_attack(
        n=crypto_result["small_rsa"]["n"],
        e=crypto_result["small_rsa"]["e"],
        ciphertext=crypto_result["small_ciphertext"],
    )



    # Step 3: Grover's search simulation
    grover_result = grover_search_simulation()



    return {
        "crypto": crypto_result,
        "shor": shor_result,
        "grover": grover_result,
    }
