"""
loads gridworld config from .env at project root, with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

ROWS, COLS = int(os.getenv("ROWS", 3)), int(os.getenv("COLS", 4))
N_STATES = ROWS * COLS
N_ACTIONS = int(os.getenv("N_ACTIONS", 4))

GOAL = int(os.getenv("GOAL", 3))
DANGER = int(os.getenv("DANGER", 7))
WALL = int(os.getenv("WALL", 5))
START = int(os.getenv("START", 8))

GOAL_REWARD = float(os.getenv("GOAL_REWARD", "1.0"))
DANGER_REWARD = float(os.getenv("DANGER_REWARD", "-1.0"))
STEP_REWARD = float(os.getenv("STEP_REWARD", "-0.1"))
MAX_STEPS = int(os.getenv("MAX_STEPS", 30))

N_EP  = int(os.getenv("N_EP", 8000))
GAMMA = float(os.getenv("GAMMA", 0.95))

ARROW = {0: (0, -0.3), 
         1: (0.3, 0), 
         2: (-0.3, 0), 
         3: (0, 0.3)}

ACTIONS = {
    0: (-1,  0), 
    1: ( 0,  1), 
    2: ( 0, -1), 
    3: ( 1,  0), 
}

ACTION_NAMES = {
    0: "up",
    1: "right",
    2: "left",
    3: "down",
}

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "docs/02-results")
