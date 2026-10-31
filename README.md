# BB84 & Q-Learning Simulator

An educational quantum-key-distribution simulation with a Q-learning eavesdropper, interception actions, QBER-related state/rewards, encryption experiments and classical/quantum attack demonstrations.

## Source layout

- `bb84_environment.py`: BB84 episode environment and measurement model.
- `eve_agent.py`, `q_learning.py`: tabular learning agent and training loop.
- `simulation.py`, `main.py`: simulation and interactive experiment interface.
- `bb84_qkd.py`: basic encoding, measurement and key sifting.
- `encryption.py`, `phase1_classical_crypto.py`: classical encryption demonstrations.
- `quantum_attack_simulation.py`, `phase3_pqc_demo.py`: optional attack/PQC experiments.

## Setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

The interactive workflow may run long training/attack experiments. Generated Q-tables and keys stay local. PQC-specific demo backends are optional and version-dependent; inspect `phase3_pqc_demo.py` and install a compatible backend separately if you want that demonstration. Source imports and a small environment/AES smoke check are recorded in `PUBLICATION.md`.

## Limits

This is a simulation, not a production cryptographic implementation or a claim of quantum-security advantage. Toy search spaces and attack timings do not generalize to real deployed cryptography. Simulation-generated keys and demonstration encryption must not protect real data. No physical quantum channel, authenticated key exchange or production security audit is provided.

The publication contains the simulation source only; seminar papers, presentation artifacts, saved output and learned Q-tables are excluded.
