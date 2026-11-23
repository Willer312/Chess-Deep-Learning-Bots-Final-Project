from pathlib import Path
import chess
import torch
import torch.nn as nn
from core.base_bot import BaseBot
from core.board_encoder import to_tensor_planes
from core.move_vocab import ACTION_SPACE_SIZE, select_best_legal_move


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class PolicyResNet(nn.Module):
    def __init__(self, in_channels: int = 18, channels: int = 128, num_blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, ACTION_SPACE_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.policy_head(x)


class PolicyCNNBot(BaseBot):
    def __init__(self, name: str = "Policy ResNet", weights_path: str = "weights/policy_resnet.pt"):
        super().__init__(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyResNet().to(self.device)
        if Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        self.model.eval()

    def _choose_move(self, board: chess.Board, time_limit_ms: int) -> tuple[chess.Move, float, int]:
        tensor = to_tensor_planes(board, self.device)
        with torch.no_grad():
            logits = self.model(tensor).squeeze(0)
        move, confidence = select_best_legal_move(board, logits)
        win_prob = 0.5 + (confidence - 0.5) * 0.2 if board.turn == chess.WHITE else 0.5 - (confidence - 0.5) * 0.2
        return move, win_prob, 1