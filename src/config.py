"""
config.py — Central configuration for all hyperparameters and paths.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH       = os.path.join(BASE_DIR, "data", "EV.csv")
FIGURES_DIR     = os.path.join(BASE_DIR, "figures_output")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Data split (chronological) ─────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15 (remainder)

TEST_START_DATE = "2025-05-01"   # held-out test set start (paper: May 1–Dec 15 2025)

# ── Sequence / features ────────────────────────────────────────────────────
LOOKBACK  = 30          # days
HORIZON   = 1           # single-step daily forecast
FEATURES  = 15          # total input features (matches paper Table I)

WEATHER_FEATURES = ["tmpf", "relh", "feel", "sped", "p01m", "snowdepth"]
TARGET           = "Energy Provided (kWh)"

# ── Transformer-LSTM architecture ──────────────────────────────────────────
D_MODEL      = 64
N_HEADS      = 4
D_FF         = 256
N_LAYERS     = 2          # stacked encoder blocks
LSTM_HIDDEN  = 128
DROPOUT      = 0.10

# ── Training ───────────────────────────────────────────────────────────────
EPOCHS        = 150
BATCH_SIZE    = 64
LR            = 1e-3
PATIENCE      = 10        # early stopping
RANDOM_SEED   = 42

# ── Baseline hyperparameters ───────────────────────────────────────────────
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH    = 15

# ── MILP ───────────────────────────────────────────────────────────────────
MILP_ALPHA     = 0.5   # cost weight
MILP_BETA      = 0.3   # peak weight
MILP_GAMMA     = 0.2   # wait-time weight
P_MAX_KW       = 50.0  # charger capacity limit (kW)
CHARGE_RATE_KW = 7.2   # per-EV charge rate (kW)
N_SLOTS        = 96    # 15-min slots per day
N_EVS          = 20    # simulated EVs for the MILP demo

# ── Figures ────────────────────────────────────────────────────────────────
DPI       = 300
FIG_EXT   = "png"
FONT_SIZE = 11
