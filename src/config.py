# =========================================
# DATA CONFIGURATION
# =========================================

DATE_COL = "DATE"
TIME_COL = "TIME"
SYMBOL_COL = "SYMBOL"
DATETIME_COL = "DATETIME"

PRICE_COL = "MID_OPEN"
TARGET_COL = "TARGET_RETURN"


# =========================================
# FEATURE CONFIGURATION
# =========================================

FEATURE_COLS = [

    # Current information (SUM_DELTA_CURRENT is the demand aggregate of bar t-1;
    # see src/preprocessing/features.py for why it must be lagged)
    "RETURN_CURRENT",
    "SUM_DELTA_CURRENT",

    # Lagged returns
    "RETURN_LAG_1",
    "RETURN_LAG_2",
    "RETURN_LAG_3",

    # Lagged order flow
    "SUM_DELTA_LAG_1",
    "SUM_DELTA_LAG_2",
    "SUM_DELTA_LAG_3",

    # Rolling return statistics
    "ROLLING_RETURN_MEAN_3",
    "ROLLING_RETURN_MEAN_5",

    "ROLLING_RETURN_STD_3",
    "ROLLING_RETURN_STD_5",

    # Rolling order flow statistics
    "ROLLING_SUM_DELTA_MEAN_3",
    "ROLLING_SUM_DELTA_MEAN_5",

    # Cross-sectional features
    "RETURN_CS_RANK",
    "SUM_DELTA_CS_RANK",

    # Standardized features
    "SUM_DELTA_ZSCORE",

    # Intraday seasonality
    "MINUTES_FROM_OPEN",
    "INTRADAY_TIME_FRACTION",
    "TIME_SIN",
    "TIME_COS"
]


# =========================================
# BAR TIMING
# =========================================

# The session runs 09:30 to 15:40 in 10-minute bars.
SESSION_MINUTES = 370
BARS_PER_DAY = 38
TRADING_DAYS_PER_YEAR = 252

# Used for every Sharpe annualisation in the project. Report the per-bar Sharpe
# alongside it: sqrt(9576) = 97.9 makes the annualised figure large by
# construction over a short test window.
PERIODS_PER_YEAR = TRADING_DAYS_PER_YEAR * BARS_PER_DAY


# =========================================
# PREPROCESSING PARAMETERS
# =========================================

MIN_TIMESTAMPS_PER_SYMBOL = 30

RETURN_CLIP_LOWER_QUANTILE = 0.01
RETURN_CLIP_UPPER_QUANTILE = 0.99

TARGET_HORIZON = 1


# =========================================
# TRAIN / VALIDATION / TEST SPLIT
# =========================================

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# =========================================
# WALK-FORWARD VALIDATION
# =========================================

TRAIN_WINDOW = 100_000
TEST_WINDOW = 20_000

STEP_SIZE = 20_000


# =========================================
# LSTM CONFIGURATION
# =========================================

SEQUENCE_LENGTH = 36

ATTENTION_SEQUENCE_LENGTH = 12

LSTM_HIDDEN_SIZE = 128

LSTM_NUM_LAYERS = 1

LSTM_DROPOUT = 0.20

BATCH_SIZE = 256

LEARNING_RATE = 1e-3

NUM_EPOCHS = 20


# =========================================
# PORTFOLIO CONSTRUCTION
# =========================================

LONG_QUANTILE = 0.10
SHORT_QUANTILE = 0.10

TRANSACTION_COST = 0.0005


# =========================================
# RANDOM SEED
# =========================================

RANDOM_STATE = 42