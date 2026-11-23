# Deep Learning Chess Tournament & Interactive Arena

A modular FastAPI + `chessboard.js` tournament arena comparing distinct deep learning chess architectures trained on curated Lichess Elite (2400+ vs. 2200+ Elo) games.

## Implemented Architectures
1. **Random Baseline (`bots/random_bot.py`):** Uniform random legal move selector.
2. **Bot 1 — Policy ResNet (`bots/policy_cnn.py`):** 14-layer residual convolutional network trained via supervised imitation learning over `18x8x8` board planes (`4,672` move logits). Searchless (`1` node/move).
3. **Bot 2 — ValueMLP + Alpha-Beta Search (`bots/value_search.py`):** Residual MLP value network (`769 -> 512 -> 512 -> 1`) trained on game outcomes (`MSELoss`), paired with batched GPU Depth-3 Alpha-Beta search and endgame progress heuristics.
4. **Bot 3 — Searchless Chess Transformer *(In Progress)*:** Self-attention sequence encoder over 69 board state tokens.
5. **Bot 4 — AlphaZero-Style Dual-Headed Net + MCTS *(Planned)*:** Joint policy-value network guided by PUCT Monte Carlo Tree Search.

## Setup & Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch fastapi uvicorn python-chess numpy