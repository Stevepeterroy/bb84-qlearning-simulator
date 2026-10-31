

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

NUM_ACTIONS = 4            
QBER_THRESHOLD = 0.20      


PROGRESS_BINS = 10        
QBER_BINS = 5              
INTERCEPT_RATIO_BINS = 5   
LAST_ACTION_STATES = NUM_ACTIONS + 1  
BASIS_MATCH_BINS = 3       

# Quantum channel noise
NOISE_PROBABILITY = 0.02   

# Reward structure — balanced multi-objective (stealth-focused)
REWARD_CORRECT_BIT = 2.0        # Eve correctly predicts Alice's bit (info_gain * 2)
REWARD_STEALTH_BONUS = 5.0      # Bonus: low QBER (<15%) with some leakage
REWARD_SKIP_BONUS = 0.5         # Small reward for restraint (stealth)
REWARD_DETECTED = -20.0         # QBER exceeds threshold — heavy penalty
REWARD_OVER_INTERCEPT = -5.0    # Penalty if interception rate > 20%
REWARD_HARD_QBER_PENALTY = -25.0  # Extra penalty if QBER > 25%
REWARD_ACTION_INTERCEPT = -0.5  # Per-action cost for intercepting
REWARD_QBER_SCALE = -12.0       # Continuous QBER penalty scale


class BB84Environment:
    
    def __init__(self, n_qubits: int = 200):
       
        self.n_qubits = n_qubits
        self.simulator = AerSimulator()

        # 5-dimensional state space
        self.n_progress_states = PROGRESS_BINS
        self.n_qber_states = QBER_BINS + 1            # 0 to QBER_BINS inclusive
        self.n_intercept_states = INTERCEPT_RATIO_BINS
        self.n_last_action_states = LAST_ACTION_STATES
        self.n_basis_match_states = BASIS_MATCH_BINS

        self.state_space_size = (
            self.n_progress_states *
            self.n_qber_states *
            self.n_intercept_states *
            self.n_last_action_states *
            self.n_basis_match_states
        )
        self.action_space_size = NUM_ACTIONS

        self.reset()

    def reset(self) -> int:
        """
        Reset the environment for a new episode.

        Generates new Alice bits/bases and Bob bases.

        Returns:
            Initial state index.
        """
        # Alice generates random bits and bases
        self.alice_bits = np.random.randint(0, 2, size=self.n_qubits)
        self.alice_bases = np.random.randint(0, 2, size=self.n_qubits)

        # Bob generates random measurement bases
        self.bob_bases = np.random.randint(0, 2, size=self.n_qubits)

        # Encode Alice's qubits
        self.alice_circuits = self._encode_qubits(self.alice_bits, self.alice_bases)

        # Tracking
        self.current_qubit = 0
        self.bob_results = np.zeros(self.n_qubits, dtype=int)
        self.eve_predictions = np.full(self.n_qubits, -1, dtype=int)
        self.intercepted = np.zeros(self.n_qubits, dtype=bool)
        self.total_errors = 0
        self.total_compared = 0
        self.n_intercepted_so_far = 0
        self.done = False
        self.last_action = NUM_ACTIONS  # Initial "no action" state
        self.eve_info_gained = 0
        self.episode_reward = 0.0

        # Basis match tracking for state feature
        self.basis_matches_seen = 0
        self.basis_total_seen = 0

        return self._get_state_index()

    def step(self, action: int) -> tuple:
        
        if self.done or self.current_qubit >= self.n_qubits:
            return self._get_state_index(), 0.0, True, {}

        i = self.current_qubit
        reward = 0.0
        eve_measured = False
        eve_bit = -1

        # ── Execute Eve's action ─────────────────────────────────
        if action == 0:
            # Skip — no interception; small stealth reward
            circuit_for_bob = self.alice_circuits[i]
            reward = REWARD_SKIP_BONUS

        elif action == 1:
            # Intercept in Z basis
            circuit_for_bob, eve_bit = self._eve_intercept(i, eve_basis=0)
            eve_measured = True

        elif action == 2:
            # Intercept in X basis
            circuit_for_bob, eve_bit = self._eve_intercept(i, eve_basis=1)
            eve_measured = True

        elif action == 3:
            # Adaptive partial intercept — low probability (0.1–0.3)
            intercept_prob = np.random.uniform(0.1, 0.3)
            if np.random.random() < intercept_prob:
                eve_basis = np.random.randint(0, 2)
                circuit_for_bob, eve_bit = self._eve_intercept(i, eve_basis)
                eve_measured = True
            else:
                circuit_for_bob = self.alice_circuits[i]
                reward = REWARD_SKIP_BONUS

        # ── Bob measures (with quantum channel noise) ────────────
        bob_bit = self._bob_measure(circuit_for_bob, self.bob_bases[i])
        self.bob_results[i] = bob_bit

        # ── Track interception and predictions ───────────────────
        if eve_measured:
            self.intercepted[i] = True
            self.n_intercepted_so_far += 1
            self.eve_predictions[i] = eve_bit

            # Check if Eve correctly predicted Alice's bit
            if eve_bit == self.alice_bits[i]:
                reward += REWARD_CORRECT_BIT
                self.eve_info_gained += 1

        # ── Update basis match tracking ──────────────────────────
        self.basis_total_seen += 1
        if self.alice_bases[i] == self.bob_bases[i]:
            self.basis_matches_seen += 1

        # ── Update QBER (only for matching bases) ────────────────
        if self.alice_bases[i] == self.bob_bases[i]:
            self.total_compared += 1
            if self.alice_bits[i] != bob_bit:
                self.total_errors += 1

        current_qber = self._get_qber()
        current_intercept_rate = self.n_intercepted_so_far / (i + 1)

        # ── Reward shaping: multi-objective stealth reward ────────

        # Continuous QBER penalty (always applied)
        if self.total_compared >= 3:
            reward += current_qber * REWARD_QBER_SCALE

        # Per-action intercept cost (discourages unnecessary interception)
        if eve_measured:
            reward += REWARD_ACTION_INTERCEPT

        if current_qber > QBER_THRESHOLD and self.total_compared >= 5:
            # Detected — heavy penalty, episode terminates
            reward += REWARD_DETECTED
            self.done = True

        # Hard QBER constraint — extra punishment above 25%
        if current_qber > 0.25 and self.total_compared >= 5:
            reward += REWARD_HARD_QBER_PENALTY

        # Stealth bonus: low QBER + some info gained
        if eve_measured and current_qber < 0.15 and self.eve_info_gained > 0:
            reward += REWARD_STEALTH_BONUS

        # Penalty for aggressive interception (rate > 20%)
        if current_intercept_rate > 0.20 and (i + 1) >= 10:
            reward += REWARD_OVER_INTERCEPT

        # ── Advance to next qubit ────────────────────────────────
        self.last_action = action
        self.episode_reward += reward
        self.current_qubit += 1

        if self.current_qubit >= self.n_qubits:
            self.done = True

        next_state = self._get_state_index()
        info = {
            "qber": current_qber,
            "eve_bit": eve_bit,
            "alice_bit": int(self.alice_bits[i]),
            "bob_bit": bob_bit,
            "intercepted": eve_measured,
            "correct_prediction": eve_bit == self.alice_bits[i] if eve_measured else False,
            "interception_rate": current_intercept_rate,
        }

        return next_state, reward, self.done, info

    # ─────────────────── Quantum Operations ──────────────────────

    def _encode_qubits(self, bits: np.ndarray,
                       bases: np.ndarray) -> list[QuantumCircuit]:
        """Encode classical bits into quantum circuits."""
        circuits = []
        for bit, base in zip(bits, bases):
            qc = QuantumCircuit(1, 1)
            if bit == 1:
                qc.x(0)
            if base == 1:
                qc.h(0)
            qc.barrier()
            circuits.append(qc)
        return circuits

    def _eve_intercept(self, qubit_idx: int,
                       eve_basis: int) -> tuple[QuantumCircuit, int]:
        """
        Eve intercepts a qubit: measure in chosen basis, re-encode, forward.

        Returns:
            (new_circuit_for_bob, eve_measured_bit)
        """
        # Measure in Eve's basis
        eve_qc = self.alice_circuits[qubit_idx].copy()
        if eve_basis == 1:
            eve_qc.h(0)
        eve_qc.measure(0, 0)

        job = self.simulator.run(eve_qc, shots=1)
        counts = job.result().get_counts()
        eve_bit = int(list(counts.keys())[0])

        # Re-encode in Eve's basis
        new_qc = QuantumCircuit(1, 1)
        if eve_bit == 1:
            new_qc.x(0)
        if eve_basis == 1:
            new_qc.h(0)
        new_qc.barrier()

        return new_qc, eve_bit

    def _bob_measure(self, circuit: QuantumCircuit, bob_basis: int) -> int:
        """
        Bob measures a qubit in his chosen basis.

        Includes quantum channel noise: with probability NOISE_PROBABILITY,
        the measured bit is flipped to simulate natural channel errors.
        """
        meas_qc = circuit.copy()
        if bob_basis == 1:
            meas_qc.h(0)
        meas_qc.measure(0, 0)

        job = self.simulator.run(meas_qc, shots=1)
        counts = job.result().get_counts()
        measured_bit = int(list(counts.keys())[0])

        # Quantum channel noise simulation
        if np.random.random() < NOISE_PROBABILITY:
            measured_bit = 1 - measured_bit

        return measured_bit

    # ─────────────────── State Helpers ───────────────────────────

    def _get_qber(self) -> float:
        """Calculate current QBER."""
        if self.total_compared == 0:
            return 0.0
        return self.total_errors / self.total_compared

    def _discretize_qber(self, qber: float) -> int:
        """Discretize QBER into bins for Q-table indexing."""
        return min(int(qber * QBER_BINS), QBER_BINS)

    def _discretize_intercept_ratio(self) -> int:
        """Discretize current interception ratio into bins."""
        if self.current_qubit == 0:
            return 0
        ratio = self.n_intercepted_so_far / self.current_qubit
        return min(int(ratio * INTERCEPT_RATIO_BINS), INTERCEPT_RATIO_BINS - 1)

    def _discretize_basis_match_prob(self) -> int:
        """
        Discretize estimated basis match probability into bins.

        Returns 0 (low), 1 (medium), or 2 (high).
        Theoretical: 50% match rate for random bases.
        """
        if self.basis_total_seen == 0:
            return 1  # Default to medium (0.5 expected)
        prob = self.basis_matches_seen / self.basis_total_seen
        if prob < 0.40:
            return 0  # Low
        elif prob < 0.60:
            return 1  # Medium (expected ~0.5)
        else:
            return 2  # High

    def _get_state_index(self) -> int:
        """
        Convert 5-dimensional state to flat index for Q-table lookup.

        Dimensions: (progress, qber, intercept_ratio, last_action, basis_match)
        """
        qi = min(self.current_qubit, self.n_qubits - 1)
        progress_bin = min(int(qi / self.n_qubits * PROGRESS_BINS), PROGRESS_BINS - 1)
        qber_bin = self._discretize_qber(self._get_qber())
        intercept_bin = self._discretize_intercept_ratio()
        la = self.last_action
        basis_bin = self._discretize_basis_match_prob()

        # Flatten: progress × qber × intercept × last_action × basis_match
        idx = progress_bin
        idx = idx * self.n_qber_states + qber_bin
        idx = idx * self.n_intercept_states + intercept_bin
        idx = idx * self.n_last_action_states + la
        idx = idx * self.n_basis_match_states + basis_bin

        return idx

    # ─────────────────── Post-Episode Analysis ───────────────────

    def get_sifted_keys(self) -> tuple:
        """Get sifted keys after episode completes."""
        matching = self.alice_bases == self.bob_bases
        alice_key = self.alice_bits[matching]
        bob_key = self.bob_results[matching]
        return alice_key, bob_key

    def get_episode_stats(self) -> dict:
        """Get comprehensive episode statistics."""
        alice_key, bob_key = self.get_sifted_keys()
        n_intercepted = int(np.sum(self.intercepted))
        correct_preds = sum(
            1 for i in range(self.n_qubits)
            if self.intercepted[i] and self.eve_predictions[i] == self.alice_bits[i]
        )

        # Key leakage: bits Eve correctly predicted in the sifted key
        matching = np.where(self.alice_bases == self.bob_bases)[0]
        leaked_bits = sum(
            1 for idx in matching
            if self.intercepted[idx] and self.eve_predictions[idx] == self.alice_bits[idx]
        )

        return {
            "qber": self._get_qber(),
            "sifted_key_length": len(alice_key),
            "n_intercepted": n_intercepted,
            "interception_rate": n_intercepted / self.n_qubits,
            "correct_predictions": correct_preds,
            "eve_success_rate": correct_preds / max(1, n_intercepted),
            "info_gained": self.eve_info_gained,
            "key_leakage": leaked_bits,
            "key_leakage_rate": leaked_bits / max(1, len(alice_key)),
            "total_reward": self.episode_reward,
            "detected": self._get_qber() > QBER_THRESHOLD and self.total_compared >= 5,
            "alice_key": alice_key,
            "bob_key": bob_key,
        }

    def get_eve_partial_key(self) -> list:
        """
        Extract Eve's partial key aligned with the sifted key positions.

        For each bit in the sifted key (where Alice and Bob used the same basis),
        if Eve intercepted that qubit and correctly predicted Alice's bit, the
        bit is known. Otherwise it is marked as '?' (unknown).

        Returns:
            List of length == sifted key length, where each element is
            0, 1 (known bit) or '?' (unknown bit).
        """
        matching_indices = np.where(self.alice_bases == self.bob_bases)[0]
        partial_key = []

        for idx in matching_indices:
            if self.intercepted[idx] and self.eve_predictions[idx] == self.alice_bits[idx]:
                partial_key.append(int(self.alice_bits[idx]))
            else:
                partial_key.append('?')

        return partial_key
