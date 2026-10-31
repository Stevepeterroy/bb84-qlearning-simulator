"""
bb84_qkd.py — Core BB84 Quantum Key Distribution Protocol
==========================================================

Implements the quantum portion of the BB84 protocol using Qiskit:
  - Random bit/basis generation
  - Qubit encoding via quantum circuits
  - Qubit measurement on the Aer simulator
  - Key sifting (basis reconciliation)
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ─────────────────────── Random Generation ───────────────────────────

def generate_random_bits(n: int) -> np.ndarray:
    """
    Generate n random bits (0 or 1).

    Args:
        n: Number of random bits to generate.

    Returns:
        NumPy array of n random integers in {0, 1}.
    """
    return np.random.randint(0, 2, size=n)


def generate_random_bases(n: int) -> np.ndarray:
    """
    Generate n random bases: 0 = Z (computational), 1 = X (Hadamard).

    Args:
        n: Number of random bases to generate.

    Returns:
        NumPy array of n random integers in {0, 1}.
    """
    return np.random.randint(0, 2, size=n)


# ─────────────────────── Qubit Encoding ──────────────────────────────

def encode_qubits(bits: np.ndarray, bases: np.ndarray) -> list[QuantumCircuit]:
    """
    Encode classical bits into qubits using the given bases.

    Encoding scheme:
        Z basis (base=0):  bit 0 → |0⟩,  bit 1 → |1⟩
        X basis (base=1):  bit 0 → |+⟩,  bit 1 → |−⟩

    Args:
        bits:  Array of classical bits to encode.
        bases: Array of bases (0=Z, 1=X) for encoding.

    Returns:
        List of single-qubit QuantumCircuit objects.
    """
    circuits = []
    for bit, base in zip(bits, bases):
        qc = QuantumCircuit(1, 1)

        # Encode the bit value
        if bit == 1:
            qc.x(0)  # Flip to |1⟩

        # Apply Hadamard for X basis
        if base == 1:
            qc.h(0)  # Z→X: |0⟩→|+⟩, |1⟩→|−⟩

        qc.barrier()
        circuits.append(qc)

    return circuits


# ─────────────────────── Qubit Measurement ───────────────────────────

def measure_qubits(circuits: list[QuantumCircuit],
                   bases: np.ndarray) -> np.ndarray:
    """
    Measure qubits in the given bases using the Aer simulator.

    If measuring in X basis, a Hadamard gate is applied before measurement
    to rotate back to the computational basis.

    Args:
        circuits: List of QuantumCircuit objects (one qubit each).
        bases:    Array of measurement bases (0=Z, 1=X).

    Returns:
        NumPy array of measured bits.
    """
    simulator = AerSimulator()
    results = []

    for qc, base in zip(circuits, bases):
        # Create a copy so we don't modify the original circuit
        meas_qc = qc.copy()

        # Apply Hadamard if measuring in X basis
        if base == 1:
            meas_qc.h(0)

        # Measure
        meas_qc.measure(0, 0)

        # Run on simulator (single shot)
        job = simulator.run(meas_qc, shots=1)
        counts = job.result().get_counts()

        # Extract the measured bit
        measured_bit = int(list(counts.keys())[0])
        results.append(measured_bit)

    return np.array(results)


# ─────────────────────── Key Sifting ─────────────────────────────────

def sift_key(alice_bits: np.ndarray, bob_bits: np.ndarray,
             alice_bases: np.ndarray, bob_bases: np.ndarray) -> tuple:
    """
    Sift the key: keep only bits where Alice and Bob used the same basis.

    In the BB84 protocol, Alice and Bob publicly announce their bases
    (but NOT their bit values) and discard bits where bases differ.

    Args:
        alice_bits:  Alice's original bits.
        bob_bits:    Bob's measured bits.
        alice_bases: Alice's encoding bases.
        bob_bases:   Bob's measurement bases.

    Returns:
        Tuple of (alice_sifted_key, bob_sifted_key) as NumPy arrays.
    """
    matching_indices = np.where(alice_bases == bob_bases)[0]

    alice_key = alice_bits[matching_indices]
    bob_key = bob_bits[matching_indices]

    return alice_key, bob_key
