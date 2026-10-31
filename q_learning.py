"""
q_learning.py — Q-Learning Training Loop
==========================================

Trains Eve (the RL agent) over multiple episodes of the BB84 protocol.
Tracks per-episode metrics and returns training history for plotting.
"""

import numpy as np
import sys


def train(env, agent, num_episodes: int = 5000,
          verbose: bool = True) -> dict:
    """
    Train Eve over multiple BB84 episodes using Q-Learning.

    Args:
        env:          BB84Environment instance.
        agent:        EveAgent instance.
        num_episodes: Number of training episodes.
        verbose:      Whether to print progress.

    Returns:
        Dictionary of training history arrays:
          - qber:               QBER per episode
          - reward:             Total reward per episode
          - interception_rate:  Interception rate per episode
          - success_rate:       Eve's prediction success rate per episode
          - info_gain:          Information gained per episode
          - key_leakage_rate:   Key leakage rate per episode
          - epsilon:            Exploration rate per episode
          - detected:           Whether Eve was detected per episode
          - policy_snapshots:   Policy distribution snapshots
    """
    # History tracking
    history = {
        "qber": [],
        "reward": [],
        "interception_rate": [],
        "success_rate": [],
        "info_gain": [],
        "key_leakage_rate": [],
        "epsilon": [],
        "detected": [],
        "policy_snapshots": [],
    }

    # Adaptive display interval based on total episodes
    display_interval = max(50, num_episodes // 20)
    snapshot_interval = max(50, num_episodes // 20)

    if verbose:
        print(f"\n  ╔══════════════════════════════════════════════════╗")
        ep_str = f"{num_episodes:,}"
        print(f"  ║   TRAINING EVE — Q-LEARNING ({ep_str} episodes)     ║")
        print(f"  ╚══════════════════════════════════════════════════╝\n")

    for episode in range(num_episodes):
        state = env.reset()
        episode_td_errors = []

        # ── Run one full episode ─────────────────────────────────
        while not env.done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            td_error = agent.update(state, action, reward, next_state, done)
            episode_td_errors.append(td_error)

            state = next_state

        # ── Collect episode statistics ───────────────────────────
        stats = env.get_episode_stats()

        history["qber"].append(stats["qber"])
        history["reward"].append(stats["total_reward"])
        history["interception_rate"].append(stats["interception_rate"])
        history["success_rate"].append(stats["eve_success_rate"])
        history["info_gain"].append(stats["info_gained"])
        history["key_leakage_rate"].append(stats["key_leakage_rate"])
        history["epsilon"].append(agent.epsilon)
        history["detected"].append(stats["detected"])

        # Snapshot policy periodically
        if episode % snapshot_interval == 0:
            history["policy_snapshots"].append({
                "episode": episode,
                "distribution": agent.get_policy_distribution().copy(),
            })

        # ── Decay exploration ────────────────────────────────────
        agent.decay_epsilon()

        # ── Progress display ─────────────────────────────────────
        if verbose and (episode + 1) % display_interval == 0:
            recent_n = min(display_interval, len(history["qber"]))
            recent_qber = np.mean(history["qber"][-recent_n:])
            recent_reward = np.mean(history["reward"][-recent_n:])
            recent_intercept = np.mean(history["interception_rate"][-recent_n:])
            recent_detected = np.mean(history["detected"][-recent_n:])
            recent_leakage = np.mean(history["key_leakage_rate"][-recent_n:])

            print(f"  Episode {episode + 1:>5}/{num_episodes}  │  "
                  f"ε={agent.epsilon:.4f}  │  "
                  f"QBER={recent_qber:.3f}  │  "
                  f"Reward={recent_reward:>7.1f}  │  "
                  f"Intercept={recent_intercept:.2f}  │  "
                  f"Leak={recent_leakage:.2f}  │  "
                  f"Det={recent_detected:.0%}")

    # Final policy snapshot
    history["policy_snapshots"].append({
        "episode": num_episodes,
        "distribution": agent.get_policy_distribution().copy(),
    })

    if verbose:
        print(f"\n  Training complete. Total Q-table updates: {agent.total_updates:,}")
        _print_training_summary(history, agent)

    # Convert lists to numpy arrays
    for key in history:
        if key != "policy_snapshots":
            history[key] = np.array(history[key])

    return history


def _print_training_summary(history: dict, agent) -> None:
    """Print a summary of the training results."""
    n = len(history["qber"])
    last_n = min(100, n)
    last_slice = slice(-last_n, None)

    print(f"\n  ── Training Summary ──────────────────────────────")
    print(f"  Final epsilon        : {agent.epsilon:.4f}")
    print(f"  Avg QBER (last {last_n})  : {np.mean(history['qber'][last_slice]):.4f}")
    print(f"  Avg Reward (last {last_n}): {np.mean(history['reward'][last_slice]):.2f}")
    print(f"  Avg Intercept Rate   : {np.mean(history['interception_rate'][last_slice]):.3f}")
    print(f"  Avg Success Rate     : {np.mean(history['success_rate'][last_slice]):.3f}")
    print(f"  Avg Key Leakage Rate : {np.mean(history['key_leakage_rate'][last_slice]):.3f}")
    print(f"  Detection Rate       : {np.mean(history['detected'][last_slice]):.1%}")

    # Policy distribution
    policy = agent.get_policy_distribution()
    labels = agent.get_action_labels()
    print(f"\n  ── Learned Policy Distribution ───────────────────")
    for label, frac in zip(labels, policy):
        bar = "█" * int(frac * 40)
        print(f"  {label:<16} : {frac:>5.1%}  {bar}")
