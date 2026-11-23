import chess
import numpy as np
import torch

PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

PIECE_TO_TOKEN = {
    None: 0,
    (chess.PAWN, chess.WHITE): 1,
    (chess.KNIGHT, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3,
    (chess.ROOK, chess.WHITE): 4,
    (chess.QUEEN, chess.WHITE): 5,
    (chess.KING, chess.WHITE): 6,
    (chess.PAWN, chess.BLACK): 7,
    (chess.KNIGHT, chess.BLACK): 8,
    (chess.BISHOP, chess.BLACK): 9,
    (chess.ROOK, chess.BLACK): 10,
    (chess.QUEEN, chess.BLACK): 11,
    (chess.KING, chess.BLACK): 12,
}


def encode_board_planes(board: chess.Board) -> np.ndarray:
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        ch = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        planes[ch, rank, file] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        planes[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0

    if board.turn == chess.WHITE:
        planes[16, :, :] = 1.0

    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        planes[17, ep_rank, ep_file] = 1.0

    return planes


def encode_board_tokens(board: chess.Board) -> np.ndarray:
    tokens = np.zeros(69, dtype=np.int64)
    for sq in range(64):
        piece = board.piece_at(sq)
        key = None if piece is None else (piece.piece_type, piece.color)
        tokens[sq] = PIECE_TO_TOKEN[key]

    tokens[64] = 13 if board.turn == chess.WHITE else 14
    tokens[65] = 15 if board.has_kingside_castling_rights(chess.WHITE) else 16
    tokens[66] = 17 if board.has_queenside_castling_rights(chess.WHITE) else 18
    tokens[67] = 19 if board.has_kingside_castling_rights(chess.BLACK) else 20
    tokens[68] = 21 if board.has_queenside_castling_rights(chess.BLACK) else 22
    return tokens


def encode_board_flat(board: chess.Board) -> np.ndarray:
    planes = encode_board_planes(board)
    return planes[:12].reshape(-1)


def to_tensor_planes(board: chess.Board, device: torch.device) -> torch.Tensor:
    arr = encode_board_planes(board)
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def to_tensor_tokens(board: chess.Board, device: torch.device) -> torch.Tensor:
    arr = encode_board_tokens(board)
    return torch.from_numpy(arr).unsqueeze(0).to(device)