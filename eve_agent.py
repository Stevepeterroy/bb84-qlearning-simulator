"""
eve_agent.py — Q-Learning Agent for Eavesdropper Eve
=====================================================

Implements Eve as a reinforcement learning agent using tabular Q-learning.
Eve learns to intercept qubits strategically while minimizing detection.
"""

import numpy as np
import os


class EveAgent:
    """
    Q-Learning agent for the BB84 eavesdropper (Eve).

    Uses a Q-table to learn optimal interception strategies.
    Employs epsilon-greedy exploration with decay.
    Supports persistent learning via Q-table save/load.
    """

    def __init__(self, state_space_size: int, action_space_size: int,
                 alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        Initialize the Q-Learning agent.

        Args:
            state_space_size:  Number of discrete states.
            action_space_size: Number of possible actions.
            alpha:             Learning rate.
            gamma:             Discount factor.
            epsilon:           Initial exploration rate.
            epsilon_min:       Minimum exploration rate.
            epsilon_decay:     Multiplicative decay per episode.
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table with small random values
        self.initialize_q_table()

        # Statistics
        self.total_updates = 0

    def initialize_q_table(self) -> None:
        """Initialize Q-table with small random values."""
        self.q_table = np.random.uniform(
            low=-0.01, high=0.01,
            size=(self.state_space_size, self.action_space_size)
        )

    def save_q_table(self, filepath: str = "q_table.npy") -> None:
        """
        Save the Q-table to a .npy file for persistent learning.

        Args:
            filepath: Path to save the Q-table file.
        """
        np.save(filepath, self.q_table)
        print(f"  ✓  Q-table saved to: {filepath}")

    def load_q_table(self, filepath: str = "q_table.npy") -> bool:
        """
        Load a previously saved Q-table from a .npy file.

        Validates shape, clips extreme values, and applies partial reset
        to prevent bad bias from previous training.

        Args:
            filepath: Path to load the Q-table from.

        Returns:
            True if Q-table was loaded successfully, False otherwise.
        """
        if not os.path.exists(filepath):
            print(f"  ⚠  No existing Q-table found at: {filepath}")
            print(f"     Starting with fresh Q-table")
            return False

        loaded = np.load(filepath)
        expected_shape = (self.state_space_size, self.action_space_size)

        if loaded.shape != expected_shape:
            print(f"  ⚠  Q-table shape mismatch!")
            print(f"     Loaded: {loaded.shape}  Expected: {expected_shape}")
            print(f"     Reinitializing Q-table from scratch")
            self.initialize_q_table()
            return False

        # Clip extreme values to prevent runaway bias
        loaded = np.clip(loaded, -50, 50)

        # Partial reset: blend loaded knowledge with fresh randomness
        # to remove bad learned bias while retaining useful patterns
        random_q = np.random.uniform(-1, 1, loaded.shape)
        self.q_table = 0.7 * loaded + 0.3 * random_q

        print(f"  ✓  Loaded existing Q-table from: {filepath}")
        print(f"     Shape: {loaded.shape}  Non-zero entries: {np.count_nonzero(loaded):,}")
        print(f"     Applied: clip[-50,50] + partial reset (70% loaded + 30% random)")
        return True

    def choose_action(self, state: int) -> int:
        """
        Choose an action using epsilon-greedy policy with tie-breaking.

        With probability epsilon, choose a random action (exploration).
        Otherwise, choose the action with the highest Q-value.
        Ties are broken randomly to avoid deterministic bias.

        Args:
            state: Current state index.

        Returns:
            Selected action index.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space_size)
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            # Random tie-breaking among all actions with max Q-value
            best_actions = np.where(q_values == max_q)[0]
            return int(np.random.choice(best_actions))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool = False) -> float:
        """
        Update Q-table using the Q-learning update rule.

        Q(s,a) = Q(s,a) + α * (reward + γ * max(Q(s',a')) - Q(s,a))

        Args:
            state:      Current state index.
            action:     Action taken.
            reward:     Reward received.
            next_state: Next state index.
            done:       Whether the episode has ended.

        Returns:
            The TD error (for monitoring convergence).
        """
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

        self.total_updates += 1
        return td_error

    def decay_epsilon(self) -> None:
        """Decay the exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy_distribution(self) -> np.ndarray:
        """
        Get the distribution of preferred actions across all states.

        Returns:
            Array of shape (action_space_size,) with the fraction of
            states that prefer each action.
        """
        best_actions = np.argmax(self.q_table, axis=1)
        distribution = np.zeros(self.action_space_size)
        for a in range(self.action_space_size):
            distribution[a] = np.sum(best_actions == a) / len(best_actions)
        return distribution

    def get_action_labels(self) -> list[str]:
        """Get human-readable labels for each action."""
        return ["Skip", "Intercept (Z)", "Intercept (X)", "Partial"]

    def reconstruct_partial_key(self, env) -> tuple[str, list]:
        """
        Reconstruct Eve's partial key from the environment state.

        Uses the environment's get_eve_partial_key() to build a
        human-readable string and a list of known/unknown bits.

        Args:
            env: BB84Environment instance (after episode completion).

        Returns:
            Tuple of (partial_key_string, partial_key_list).
            The string uses '?' for unknown bits, e.g. "1?10?100??11".
            The list uses '?' for unknown and int(0/1) for known bits.
        """
        partial_key_list = env.get_eve_partial_key()
        partial_key_str = ''.join(str(b) for b in partial_key_list)
        return partial_key_str, partial_key_list

    def __repr__(self) -> str:
        return (f"EveAgent(states={self.state_space_size}, "
                f"actions={self.action_space_size}, "
                f"ε={self.epsilon:.4f}, "
                f"updates={self.total_updates})")
