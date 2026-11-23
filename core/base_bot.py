from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
import chess


@dataclass
class MoveResult:
    move: chess.Move
    win_probability: float
    nodes_evaluated: int   
    latency_ms: float      


class BaseBot(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def _choose_move(self, board: chess.Board, time_limit_ms: int) -> tuple[chess.Move, float, int]:
        pass

    def select_move(self, board: chess.Board, time_limit_ms: int = 1000) -> MoveResult:
        start_time = time.perf_counter()
        move, win_prob, nodes = self._choose_move(board, time_limit_ms)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if move not in board.legal_moves:
            raise ValueError(f"{self.name} produced illegal move {move} on FEN {board.fen()}")

        return MoveResult(
            move=move,
            win_probability=float(max(0.0, min(1.0, win_prob))),
            nodes_evaluated=int(nodes),
            latency_ms=round(latency_ms, 2),
        )