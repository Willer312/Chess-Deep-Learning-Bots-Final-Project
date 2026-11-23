import chess
import numpy as np
import torch

QUEEN_DIRS = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]

KNIGHT_DELTAS = [
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
]

UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
UNDERPROMO_DFILES = [-1, 0, 1]

ACTION_SPACE_SIZE = 64 * 73


def move_to_index(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    from_r, from_f = chess.square_rank(from_sq), chess.square_file(from_sq)
    to_r, to_f = chess.square_rank(to_sq), chess.square_file(to_sq)
    dr = to_r - from_r
    df = to_f - from_f

    if move.promotion is not None and move.promotion != chess.QUEEN:
        p_idx = UNDERPROMO_PIECES.index(move.promotion)
        d_idx = UNDERPROMO_DFILES.index(df)
        plane = 64 + p_idx * 3 + d_idx
        return from_sq * 73 + plane

    if (dr, df) in KNIGHT_DELTAS:
        plane = 56 + KNIGHT_DELTAS.index((dr, df))
        return from_sq * 73 + plane

    dist = max(abs(dr), abs(df))
    step_r = 0 if dr == 0 else dr // abs(dr)
    step_f = 0 if df == 0 else df // abs(df)
    dir_idx = QUEEN_DIRS.index((step_r, step_f))
    plane = dir_idx * 7 + (dist - 1)
    return from_sq * 73 + plane


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask


def select_best_legal_move(board: chess.Board, logits: torch.Tensor) -> tuple[chess.Move, float]:
    legal_moves = list(board.legal_moves)
    indices = [move_to_index(m) for m in legal_moves]
    legal_logits = logits[indices]
    probs = torch.softmax(legal_logits, dim=-1)
    best_local = int(torch.argmax(probs).item())
    return legal_moves[best_local], float(probs[best_local].item())