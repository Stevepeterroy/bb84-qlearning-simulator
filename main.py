"""
main.py — Quantum Cryptography Research Simulation: Main Entry Point
=====================================================================

Two-phase research simulation:

  PHASE 1: Quantum Attack on Classical Cryptography
    - RSA/AES encryption demonstration
    - Shor's algorithm attack on RSA (simulated)
    - Grover's algorithm speedup analysis

  PHASE 2: BB84 + RL Eavesdropping Simulation
    - AES-256-GCM encryption with BB84 key distribution
    - Q-Learning RL agent (Eve) for stealth interception
    - Key leakage analysis + probabilistic key reconstruction

Usage:
    python main.py
    python main.py "Your secret message here"
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bb84_environment import BB84Environment
from eve_agent import EveAgent
from q_learning import train
from simulation import run_demo_episode, run_aes_pipeline
from quantum_attack_simulation import run_phase1
from phase3_pqc_demo import run_pqc_demo


# ─────────────────────── Configuration ───────────────────────────────

N_QUBITS = 200              # Qubits per episode (was 50 → better statistics)
NUM_EPISODES = 5000          # Training episodes (was 1000 → deeper learning)
DEFAULT_MESSAGE = "HELLO QUANTUM"   # Default message to encrypt
Q_TABLE_PATH = "q_table.npy"         # Persistent Q-table file

# Q-Learning hyperparameters
ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999        # Adjusted for 5000 episodes (reaches ~0.007 by end)


# ─────────────────────── Visualization ───────────────────────────────

def smooth(data: np.ndarray, window: int = 50) -> np.ndarray:
    """Apply simple moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_training_results(history: dict, save_path: str = "rl_bb84_training.png"):
    """
    Generate comprehensive training visualization.

    Creates a 2×3 grid:
      1. QBER vs Episode
      2. Eve Reward vs Episode
      3. Interception Rate vs Episode
      4. Policy Evolution
      5. Key Leakage Rate vs Episode

    Args:
        history:   Training history from q_learning.train().
        save_path: File path to save the plot.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("BB84 + Q-Learning: Eve's Training Progress",
                 fontsize=16, fontweight="bold", y=0.98)

    episodes = np.arange(len(history["qber"]))

    # ── Plot 1: QBER vs Episode ──────────────────────────────────
    ax = axes[0, 0]
    ax.plot(episodes, history["qber"], alpha=0.15, color="#3498db", linewidth=0.5)
    ax.plot(smooth(history["qber"]), color="#2c3e50", linewidth=2, label="Smoothed QBER")
    ax.axhline(y=0.20, color="#e74c3c", linestyle="--", linewidth=1.5,
               label="Detection Threshold (20%)")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("QBER", fontsize=11)
    ax.set_title("Quantum Bit Error Rate", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 0.5)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Plot 2: Eve Reward vs Episode ────────────────────────────
    ax = axes[0, 1]
    ax.plot(episodes, history["reward"], alpha=0.15, color="#e67e22", linewidth=0.5)
    ax.plot(smooth(history["reward"]), color="#d35400", linewidth=2, label="Smoothed Reward")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Reward", fontsize=11)
    ax.set_title("Eve's Reward per Episode", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Plot 3: Interception Rate vs Episode ─────────────────────
    ax = axes[0, 2]
    ax.plot(episodes, history["interception_rate"], alpha=0.15,
            color="#27ae60", linewidth=0.5)
    ax.plot(smooth(history["interception_rate"]), color="#1e8449",
            linewidth=2, label="Smoothed Rate")
    ax.plot(smooth(history["success_rate"]), color="#8e44ad",
            linewidth=2, linestyle="--", label="Eve Success Rate")
    ax.axhline(y=0.30, color="#e74c3c", linestyle=":", linewidth=1,
               label="Aggression Threshold (30%)")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("Interception & Success Rates", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Plot 4: Policy Evolution ─────────────────────────────────
    ax = axes[1, 0]
    snapshots = history["policy_snapshots"]
    if snapshots:
        policy_episodes = [s["episode"] for s in snapshots]
        action_labels = ["Skip", "Z-Intercept", "X-Intercept", "Partial"]
        colors = ["#95a5a6", "#3498db", "#e74c3c", "#f39c12"]

        for a_idx in range(4):
            values = [s["distribution"][a_idx] for s in snapshots]
            ax.plot(policy_episodes, values, marker="o", markersize=4,
                    linewidth=2, label=action_labels[a_idx], color=colors[a_idx])

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Policy Fraction", fontsize=11)
    ax.set_title("Eve's Policy Evolution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="center right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Plot 5: Key Leakage Rate vs Episode ──────────────────────
    ax = axes[1, 1]
    ax.plot(episodes, history["key_leakage_rate"], alpha=0.15,
            color="#e74c3c", linewidth=0.5)
    ax.plot(smooth(history["key_leakage_rate"]), color="#c0392b",
            linewidth=2, label="Smoothed Key Leakage")
    ax.axhline(y=0.0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Key Leakage Rate", fontsize=11)
    ax.set_title("Key Leakage Rate per Episode", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Plot 6: Remove empty subplot ─────────────────────────────
    axes[1, 2].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  📊 Training plots saved to: {save_path}")
    plt.close()


def plot_qber_comparison(qber_no_eve: float, qber_with_eve: float,
                         qber_rl_eve: float, save_path: str = "qber_comparison.png"):
    """
    Generate QBER comparison bar chart across three scenarios.

    Args:
        qber_no_eve:  QBER with no eavesdropper.
        qber_with_eve: QBER with random Eve.
        qber_rl_eve:  QBER with RL-trained Eve.
        save_path:    File path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    categories = ["No Eve\n(Noise Only)", "Random Eve", "RL-Trained Eve"]
    values = [qber_no_eve * 100, qber_with_eve * 100, qber_rl_eve * 100]
    colors = ["#2ecc71", "#e74c3c", "#9b59b6"]

    bars = ax.bar(categories, values, color=colors, width=0.5,
                  edgecolor="white", linewidth=2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold")

    ax.axhline(y=20, color="#e67e22", linestyle="--", linewidth=2,
               label="Detection Threshold (20%)")

    ax.set_ylabel("QBER (%)", fontsize=12)
    ax.set_title("QBER Comparison: No Eve vs Random vs RL-Trained",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(max(values) + 10, 40))
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  📊 QBER comparison plot saved to: {save_path}")
    plt.close()


def plot_quantum_attack_comparison(grover_result: dict,
                                    save_path: str = "phase1_quantum_attack_plot.png"):
    """
    Generate Phase 1 quantum attack comparison plot.

    Shows classical vs quantum attack complexity across key sizes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Classical vs Quantum Attack Complexity",
                 fontsize=14, fontweight="bold")

    # RSA vs Quantum line chart
    rsa_sizes = [1024, 2048, 3072, 4096]
    classical_years = [2.0, 300.0, 50000.0, 5000000.0]  # Approx classical time
    quantum_hours = [0.01, 0.1, 1.0, 8.0]  # Approx quantum time with Shor

    ax.semilogy(rsa_sizes, classical_years, "o-", color="#e74c3c",
                linewidth=2, markersize=8, label="Classical (years)")
    ax.semilogy(rsa_sizes, quantum_hours, "s-", color="#2ecc71",
                linewidth=2, markersize=8, label="Quantum / Shor (hours)")
    ax.fill_between(rsa_sizes, quantum_hours, classical_years,
                    alpha=0.1, color="#e74c3c")
    ax.set_xlabel("RSA Key Size (bits)", fontsize=11)
    ax.set_ylabel("Time to Break (log scale)", fontsize=11)
    ax.set_title("RSA: Classical vs Shor's Algorithm", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_key_leakage(history: dict, save_path: str = "key_leakage_plot.png"):
    """
    Generate a dedicated key leakage analysis plot.

    Shows key leakage rate and information gain over training episodes.

    Args:
        history:   Training history from q_learning.train().
        save_path: File path to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Key Leakage Analysis Over Training",
                 fontsize=14, fontweight="bold")

    episodes = np.arange(len(history["key_leakage_rate"]))

    # ── Left: Key Leakage Rate ───────────────────────────────────
    ax1.plot(episodes, history["key_leakage_rate"], alpha=0.15,
             color="#e74c3c", linewidth=0.5)
    ax1.plot(smooth(history["key_leakage_rate"]), color="#c0392b",
             linewidth=2, label="Smoothed Leakage Rate")
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Key Leakage Rate", fontsize=11)
    ax1.set_title("Key Leakage Rate vs Episode", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Right: Information Gain (bits) ───────────────────────────
    ax2.plot(episodes, history["info_gain"], alpha=0.15,
             color="#2980b9", linewidth=0.5)
    ax2.plot(smooth(history["info_gain"]), color="#1a5276",
             linewidth=2, label="Smoothed Info Gain")
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Information Gain (bits)", fontsize=11)
    ax2.set_title("Shannon Information Gain vs Episode", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


# ──────────────────── Baseline Runs ──────────────────────────────────

def run_baseline_no_eve(n_qubits: int) -> float:
    """Run BB84 without any eavesdropper and return QBER."""
    env = BB84Environment(n_qubits)
    env.reset()

    for i in range(n_qubits):
        env.step(0)  # Action 0 = skip

    stats = env.get_episode_stats()
    return stats["qber"]


def run_baseline_random_eve(n_qubits: int) -> float:
    """Run BB84 with a random eavesdropper and return QBER."""
    env = BB84Environment(n_qubits)
    env.reset()

    for i in range(n_qubits):
        action = np.random.choice([1, 2])  # Random Z or X intercept
        env.step(action)
        if env.done:
            break

    stats = env.get_episode_stats()
    return stats["qber"]


# ─────────────────────── User Input ──────────────────────────────────

def get_user_message() -> str:
    """
    Get the message to encrypt from user input or command line.

    Priority:
      1. Command-line argument: python main.py "message here"
      2. Interactive prompt
      3. Default message fallback

    Returns:
        The user's message string.
    """
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])

    print(f"\n  Enter a message to encrypt (or press Enter for default):")
    user_input = input(f"  > ").strip()

    if user_input:
        return user_input
    else:
        return DEFAULT_MESSAGE


# ─────────────────────── Main Entry Point ────────────────────────────

def main():
    """
    Run the complete two-phase quantum cryptography research simulation.

    Phase 1: Quantum attack on classical cryptography
    Phase 2: BB84 + RL eavesdropping with AES-256-GCM
    """
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   QUANTUM CRYPTOGRAPHY RESEARCH SIMULATION              ║")
    print("  ║   ──────────────────────────────────────────────────     ║")
    print("  ║   Phase 1: Quantum Attack on Classical Cryptography     ║")
    print("  ║   Phase 2: BB84 + RL Eavesdropping Simulation           ║")
    print("  ║   Phase 3: Post-Quantum Cryptography (RSA vs Kyber)     ║")
    print("  ╚══════════════════════════════════════════════════════════╝")

    # Ensure visual_datas folder exists
    os.makedirs("visual_datas", exist_ok=True)

    # Get user message
    message = get_user_message()

    # ═══════════════════════════════════════════════════════════════
    #  PHASE 1: QUANTUM ATTACK ON CLASSICAL CRYPTOGRAPHY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  ═════ PHASE 1: QUANTUM ATTACK ON CLASSICAL CRYPTOGRAPHY ═════")
    print(f"{'═' * 60}")

    phase1_result = run_phase1(message)

    # Generate Phase 1 visualization
    plot_quantum_attack_comparison(
        phase1_result["grover"],
        save_path="visual_datas/phase1_quantum_attack_plot.png"
    )

    # ═══════════════════════════════════════════════════════════════
    #  PHASE 2: BB84 + RL EAVESDROPPING SIMULATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'═' * 60}")
    print(f"  ═════ PHASE 2: BB84 + RL EAVESDROPPING SIMULATION ═════")
    print(f"{'═' * 60}")

    # Separate input for Phase 2
    print(f"\n  Enter a message for Phase 2 BB84 simulation (or press Enter for default):")
    phase2_input = input(f"  > ").strip()
    phase2_message = phase2_input if phase2_input else DEFAULT_MESSAGE

    print(f"\n  Configuration:")
    print(f"    Message            : \"{phase2_message}\"")
    print(f"    Qubits per episode : {N_QUBITS}")
    print(f"    Training episodes  : {NUM_EPISODES:,}")
    print(f"    Learning rate (α)  : {ALPHA}")
    print(f"    Discount factor (γ): {GAMMA}")
    print(f"    Epsilon: {EPSILON_START} → {EPSILON_MIN} (decay={EPSILON_DECAY})")
    print(f"    Encryption         : AES-256-GCM")
    print(f"    Channel noise      : 2% (quantum noise)")


    # ═══════════════════════════════════════════════════════════════
    # Phase 2.1: Create Environment and Agent
    # ═══════════════════════════════════════════════════════════════

    env = BB84Environment(n_qubits=N_QUBITS)
    agent = EveAgent(
        state_space_size=env.state_space_size,
        action_space_size=env.action_space_size,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    )

    print(f"\n  Environment: {env.state_space_size:,} states × {env.action_space_size} actions")
    print(f"  State dimensions: progress({env.n_progress_states}) × "
          f"qber({env.n_qber_states}) × intercept({env.n_intercept_states}) × "
          f"last_action({env.n_last_action_states}) × basis_match({env.n_basis_match_states})")
    print(f"  Agent: {agent}")

    # ── Load saved Q-table if available ────────────────────────────
    print(f"\n  Loading previous training...")
    loaded = agent.load_q_table(Q_TABLE_PATH)
    if loaded:
        # Lower epsilon for continued training (less exploration)
        agent.epsilon = max(agent.epsilon_min, 0.3)
        print(f"  Epsilon set to {agent.epsilon:.4f} for continued training")

    # ═══════════════════════════════════════════════════════════════
    # Phase 2.2: Train Eve
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  PHASE 2 — Train Eve via Q-Learning ({NUM_EPISODES:,} episodes)")
    print(f"{'═' * 60}")

    history = train(env, agent, num_episodes=NUM_EPISODES, verbose=True)

    # Save updated Q-table
    print(f"\n  Saving updated Q-table...")
    agent.save_q_table(Q_TABLE_PATH)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2.3: AES + BB84 Secure Communication Pipeline
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  PHASE 3 — AES + BB84 Secure Communication Pipeline")
    print(f"{'═' * 60}")

    pipeline_result = run_aes_pipeline(env, agent, message=phase2_message, verbose=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2.4: Baseline Comparisons
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  PHASE 4 — Baseline Comparisons")
    print(f"{'═' * 60}")

    print(f"\n  Running baselines (with quantum channel noise)...")
    qber_no_eve = run_baseline_no_eve(N_QUBITS)
    qber_random_eve = run_baseline_random_eve(N_QUBITS)
    qber_rl_eve = pipeline_result["qber"]

    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Scenario          │  QBER       │  Detected?       │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │  No Eve (noise)    │  {qber_no_eve:>8.2%}   │  {'Yes ⚠' if qber_no_eve > 0.20 else 'No  ✓':>16} │")
    print(f"  │  Random Eve        │  {qber_random_eve:>8.2%}   │  {'Yes ⚠' if qber_random_eve > 0.20 else 'No  ✓':>16} │")
    print(f"  │  RL-Trained Eve    │  {qber_rl_eve:>8.2%}   │  {'Yes ⚠' if qber_rl_eve > 0.20 else 'No  ✓':>16} │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # ═══════════════════════════════════════════════════════════════
    # Phase 2.5: Generate Plots
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  PHASE 5 — Visualization")
    print(f"{'═' * 60}")

    plot_training_results(history, save_path="visual_datas/rl_bb84_training.png")
    plot_qber_comparison(qber_no_eve, qber_random_eve, qber_rl_eve,
                         save_path="visual_datas/qber_comparison.png")
    plot_key_leakage(history, save_path="visual_datas/key_leakage_plot.png")

    # ═══════════════════════════════════════════════════════════════
    #  PHASE 3: POST-QUANTUM CRYPTOGRAPHY (RSA vs KYBER)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  ═════ PHASE 3: POST-QUANTUM CRYPTOGRAPHY (RSA vs KYBER) ═════")
    print(f"{'═' * 60}")

    phase3_result = run_pqc_demo(message)

    # ═══════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  FINAL SUMMARY")
    print(f"{'═' * 60}")

    last_n = min(100, len(history["qber"]))
    detected_pct = np.mean(history["detected"][-last_n:]) * 100
    avg_qber = np.mean(history["qber"][-last_n:])
    avg_reward = np.mean(history["reward"][-last_n:])
    avg_intercept = np.mean(history["interception_rate"][-last_n:])
    avg_leakage = np.mean(history["key_leakage_rate"][-last_n:])
    avg_success = np.mean(history["success_rate"][-last_n:])
    avg_info = np.mean(history["info_gain"][-last_n:])

    attack = pipeline_result["attack_result"]
    eve_known = pipeline_result["eve_known_bits"]
    eve_total = pipeline_result["eve_total_bits"]
    info_gain = pipeline_result["information_gain"]

    # Pre-compute formatted strings
    s_qber = f"{avg_qber:.4f}"
    s_reward = f"{avg_reward:.1f}"
    s_intercept = f"{avg_intercept:.1%}"
    s_leakage = f"{avg_leakage:.1%}"
    s_success = f"{avg_success:.1%}"
    s_info = f"{avg_info:.1f} bits"
    s_detect_pct = f"{detected_pct:.0f}%"
    s_pipe_qber = f"{pipeline_result['qber']:.4f}"
    s_detected = "Yes ⚠" if pipeline_result["detected"] else "No ✓"
    s_eve_leak = f"{eve_known}/{eve_total} bits ({eve_known/max(1,eve_total):.1%})"
    s_pipe_info = f"{info_gain:.1f} bits"
    s_eve_attack = "✓ Success" if attack["success"] else "✗ Failed"
    s_attempts = f"{attack['attempts']:,}"
    s_bob = "✓ Success" if pipeline_result["bob_success"] else "✗ Failed"

    print()
    print(f"  ┌────────────────────────────────────────────────────────────┐")
    print(f"  │  TRAINING RESULTS (last {last_n} episodes)                    │")
    print(f"  ├────────────────────────────────────────────────────────────┤")
    print(f"  │  Avg QBER              : {s_qber:<34}│")
    print(f"  │  Avg Reward            : {s_reward:<34}│")
    print(f"  │  Avg Interception Rate : {s_intercept:<34}│")
    print(f"  │  Avg Key Leakage Rate  : {s_leakage:<34}│")
    print(f"  │  Avg Eve Success Rate  : {s_success:<34}│")
    print(f"  │  Avg Information Gain  : {s_info:<34}│")

    print(f"  ├────────────────────────────────────────────────────────────┤")
    print(f"  │  PIPELINE RESULTS                                          │")
    print(f"  │  Encryption: AES-256-GCM   Noise: 2%                      │")
    print(f"  │  QBER: {s_pipe_qber:<20}  Detected: {s_detected:<18}│")
    print(f"  │  Eve Key Leakage: {s_eve_leak:<40}│")
    print(f"  │  Information Gain: {s_pipe_info:<39}│")
    print(f"  │  Eve Attack: {s_eve_attack:<20}  Attempts: {s_attempts:<13}│")
    print(f"  │  Bob Decrypt: {s_bob:<45}│")
    print(f"  ├────────────────────────────────────────────────────────────┤")
    print(f"  │  OUTPUT FILES                                              │")
    print(f"  │  rl_bb84_training.png  — Training curves (QBER/Reward)     │")
    print(f"  │  qber_comparison.png   — QBER bar chart comparison         │")
    print(f"  │  key_leakage_plot.png  — Key leakage analysis              │")
    print(f"  └────────────────────────────────────────────────────────────┘")
    print()


if __name__ == "__main__":
    main()
