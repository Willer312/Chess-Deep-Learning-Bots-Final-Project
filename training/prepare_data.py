import argparse
from pathlib import Path
import chess
import chess.pgn
import numpy as np
from core.board_encoder import encode_board_planes, encode_board_tokens
from core.move_vocab import move_to_index

RESULT_MAP = {
    "1-0": 1.0,
    "0-1": -1.0,
    "1/2-1/2": 0.0,
}


def extract_dataset(
    pgn_path: str,
    output_path: str,
    max_samples: int = 100000,
    min_elo: int = 2000,
    skip_plies: int = 6,
):
    planes_list = []
    tokens_list = []
    moves_list = []
    outcomes_list = []

    games_scanned = 0
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while len(moves_list) < max_samples:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            games_scanned += 1
            headers = game.headers
            result_str = headers.get("Result", "*")
            if result_str not in RESULT_MAP:
                continue

            if headers.get("Variant", "Standard") != "Standard":
                continue

            try:
                w_elo = int(headers.get("WhiteElo", 0))
                b_elo = int(headers.get("BlackElo", 0))
            except ValueError:
                continue

            if w_elo < min_elo or b_elo < min_elo:
                continue

            white_outcome = RESULT_MAP[result_str]
            board = game.board()

            for ply, move in enumerate(game.mainline_moves()):
                if ply >= skip_plies:
                    planes_list.append(encode_board_planes(board))
                    tokens_list.append(encode_board_tokens(board))
                    moves_list.append(move_to_index(move))
                    stm_outcome = white_outcome if board.turn == chess.WHITE else -white_outcome
                    outcomes_list.append(stm_outcome)

                    if len(moves_list) >= max_samples:
                        break

                board.push(move)

            if games_scanned % 500 == 0:
                print(f"Games scanned: {games_scanned} | Positions collected: {len(moves_list)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        planes=np.array(planes_list, dtype=np.uint8),
        tokens=np.array(tokens_list, dtype=np.uint8),
        moves=np.array(moves_list, dtype=np.int64),
        outcomes=np.array(outcomes_list, dtype=np.float32),
    )
    print(f"Saved {len(moves_list)} samples from {games_scanned} games to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/train_dataset.npz")
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--min_elo", type=int, default=2000)
    parser.add_argument("--skip_plies", type=int, default=6)
    args = parser.parse_args()

    extract_dataset(
        pgn_path=args.pgn,
        output_path=args.out,
        max_samples=args.max_samples,
        min_elo=args.min_elo,
        skip_plies=args.skip_plies,
    )