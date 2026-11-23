from pathlib import Path
import chess
import numpy as np
import torch
import torch.nn as nn
from core.base_bot import BaseBot
from core.board_encoder import encode_board_planes


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.2,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}


def encode_value_features(board: chess.Board) -> np.ndarray:
    planes = encode_board_planes(board)
    flat_pieces = planes[:12].reshape(-1)
    turn_flag = np.array([1.0 if board.turn == chess.WHITE else 0.0], dtype=np.float32)
    return np.concatenate([flat_pieces, turn_flag], axis=0)


def tactical_progress_bonus(board: chess.Board) -> float:
    w_mat = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in PIECE_VALUES.items())
    b_mat = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in PIECE_VALUES.items())
    mat_diff = w_mat - b_mat if board.turn == chess.WHITE else b_mat - w_mat
    mat_score = np.tanh(mat_diff / 8.0)

    if abs(mat_diff) >= 3.0:
        enemy_color = not board.turn
        enemy_king = board.king(enemy_color)
        friendly_king = board.king(board.turn)
        if enemy_king is not None and friendly_king is not None:
            ek_file, ek_rank = chess.square_file(enemy_king), chess.square_rank(enemy_king)
            center_dist = abs(3.5 - ek_file) + abs(3.5 - ek_rank)
            fk_file, fk_rank = chess.square_file(friendly_king), chess.square_rank(friendly_king)
            kings_dist = abs(ek_file - fk_file) + abs(ek_rank - fk_rank)
            mop_up = (center_dist / 7.0) * 0.6 + ((14.0 - kings_dist) / 14.0) * 0.4
            if mat_diff < 0:
                mop_up = -mop_up
            return float(0.8 * mat_score + 0.2 * mop_up)

    return float(mat_score)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ValueMLP(nn.Module):
    def __init__(self, in_dim: int = 769, hidden_dim: int = 512, num_blocks: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


class ValueSearchBot(BaseBot):
    def __init__(self, name: str = "Value Net + Alpha-Beta", weights_path: str = "weights/value_mlp.pt"):
        super().__init__(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ValueMLP().to(self.device)
        if Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        self.model.eval()
        self.nodes_evaluated = 0

    def _order_moves(self, board: chess.Board) -> list[chess.Move]:
        moves = list(board.legal_moves)
        moves.sort(
            key=lambda m: (
                m.promotion is not None,
                board.is_capture(m),
                board.gives_check(m),
            ),
            reverse=True,
        )
        return moves

    def _evaluate_batch(self, boards: list[chess.Board]) -> list[float]:
        if not boards:
            return []
        feats = np.stack([encode_value_features(b) for b in boards], axis=0)
        tensor = torch.from_numpy(feats).to(self.device)
        with torch.no_grad():
            nn_vals = self.model(tensor).cpu().tolist()
        blended = []
        for b, nn_v in zip(boards, nn_vals):
            bonus = tactical_progress_bonus(b)
            blended.append(0.82 * nn_v + 0.18 * bonus)
        return blended

    def _alphabeta_depth2_batched(self, board: chess.Board, alpha: float, beta: float) -> float:
        self.nodes_evaluated += 1
        if board.is_checkmate():
            return -1.0
        if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(2):
            return 0.0

        moves = self._order_moves(board)
        leaf_boards = []
        best_val = -2.0

        for move in moves:
            board.push(move)
            self.nodes_evaluated += 1
            if board.is_checkmate():
                board.pop()
                return 1.0
            if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(2):
                val = 0.0
                board.pop()
                if val > best_val:
                    best_val = val
                if best_val > alpha:
                    alpha = best_val
                if alpha >= beta:
                    return best_val
            else:
                leaf_boards.append(board.copy(stack=False))
                board.pop()

        if leaf_boards:
            opp_vals = self._evaluate_batch(leaf_boards)
            for opp_v in opp_vals:
                val = -opp_v
                if val > best_val:
                    best_val = val
                if best_val > alpha:
                    alpha = best_val
                if alpha >= beta:
                    break

        return best_val

    def _choose_move(self, board: chess.Board, time_limit_ms: int) -> tuple[chess.Move, float, int]:
        root_moves = self._order_moves(board)
        if len(root_moves) == 1:
            return root_moves[0], 0.5, 1

        self.nodes_evaluated = 0
        best_move = root_moves[0]
        alpha = -2.0
        beta = 2.0

        for move in root_moves:
            board.push(move)
            self.nodes_evaluated += 1

            if board.is_checkmate():
                board.pop()
                win_prob = 1.0 if board.turn == chess.WHITE else 0.0
                return move, win_prob, self.nodes_evaluated

            if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(2):
                score = 0.0
            else:
                score = -self._alphabeta_depth2_batched(board, -beta, -alpha)

            board.pop()

            if score > alpha:
                alpha = score
                best_move = move

        white_eval = alpha if board.turn == chess.WHITE else -alpha
        win_prob = max(0.0, min(1.0, (white_eval + 1.0) / 2.0))
        return best_move, win_prob, self.nodes_evaluated