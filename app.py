import math
from pathlib import Path
import time
import chess
import chess.engine
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from bots.random_bot import RandomBot
from bots.policy_cnn import PolicyCNNBot
from bots.value_search import ValueSearchBot

app = FastAPI(title="DL Chess Tournament Arena")
app.mount("/static", StaticFiles(directory="static"), name="static")

BOT_INSTANCES = {
    "Random Baseline": RandomBot("Random Baseline"),
    "Policy ResNet": PolicyCNNBot("Policy ResNet"),
    "Value Net + Alpha-Beta": ValueSearchBot("Value Net + Alpha-Beta"),
}

STOCKFISH_PATH = Path("stockfish/stockfish.exe")
sf_engine = None


@app.on_event("startup")
def startup_engine():
    global sf_engine
    if STOCKFISH_PATH.exists():
        sf_engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))


@app.on_event("shutdown")
def shutdown_engine():
    global sf_engine
    if sf_engine is not None:
        sf_engine.quit()
        sf_engine = None


class MoveRequest(BaseModel):
    fen: str
    bot_name: str
    time_limit_ms: int = 200


class EvalRequest(BaseModel):
    fen: str
    depth: int = 12


@app.get("/")
def read_index():
    return FileResponse("static/index.html")


@app.get("/api/bots")
def list_bots():
    return {"bots": list(BOT_INSTANCES.keys())}


@app.post("/api/eval")
def evaluate_position(req: EvalRequest):
    if sf_engine is None:
        return {"winChance": 50.0, "eval": 0.0, "mate": None}

    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    info = sf_engine.analyse(board, chess.engine.Limit(depth=req.depth))
    score = info["score"].white()

    if score.is_mate():
        mate_in = score.mate()
        win_chance = 100.0 if mate_in > 0 else 0.0
        return {"winChance": win_chance, "eval": None, "mate": mate_in}

    cp = score.score()
    win_chance = 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)
    return {"winChance": round(win_chance, 2), "eval": round(cp / 100.0, 2), "mate": None}


@app.post("/api/move")
def get_bot_move(req: MoveRequest):
    if req.bot_name not in BOT_INSTANCES:
        raise HTTPException(status_code=404, detail=f"Bot '{req.bot_name}' not found")

    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    if board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    bot = BOT_INSTANCES[req.bot_name]
    result = bot.select_move(board, req.time_limit_ms)
    move = result.move
    san = board.san(move)
    board.push(move)

    return {
        "uci": move.uci(),
        "san": san,
        "fen": board.fen(),
        "win_probability": result.win_probability,
        "nodes_evaluated": result.nodes_evaluated,
        "latency_ms": result.latency_ms,
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
    }