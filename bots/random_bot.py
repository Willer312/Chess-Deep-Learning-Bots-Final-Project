import random
import chess
from core.base_bot import BaseBot


class RandomBot(BaseBot):
    def __init__(self, name: str = "Random Baseline"):
        super().__init__(name)

    def _choose_move(self, board: chess.Board, time_limit_ms: int) -> tuple[chess.Move, float, int]:
        legal_moves = list(board.legal_moves)
        chosen_move = random.choice(legal_moves)
        return chosen_move, 0.5, 1