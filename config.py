"""
config.py — Central Configuration for Wall Street Oracle v5.0
All settings loaded from environment variables. Never hard-code secrets.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ====================== API KEYS (env-only, never hard-coded) ======================
BINANCE_TEST_API = os.getenv("BINANCE_TEST_API", "")
BINANCE_TEST_SECRET = os.getenv("BINANCE_TEST_SECRET", "")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "")

# ====================== BINANCE ENDPOINTS ======================
DEMO_BASE_URL = "https://demo-fapi.binance.com"
LIVE_BASE_URL = "https://fapi.binance.com"
BINANCE_WS_URL = "wss://fstream.binance.com/ws"
BINANCE_DATA_URL = "https://data.binance.vision"

# Use demo by default; set USE_LIVE=true in .env for real trading
USE_LIVE = os.getenv("USE_LIVE", "false").lower() == "true"
BASE_URL = LIVE_BASE_URL if USE_LIVE else DEMO_BASE_URL

# ====================== WATCHLIST ======================
DEFAULT_WATCHLIST = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'ADAUSDT', 'LINKUSDT', 'AVAXUSDT', 'DOTUSDT', 'DOGEUSDT',
    'LTCUSDT', 'BCHUSDT', 'ATOMUSDT', 'NEARUSDT', 'OPUSDT',
    'INJUSDT', 'FILUSDT', 'SEIUSDT', 'ICPUSDT',
    'AAVEUSDT', 'SNXUSDT', 'CRVUSDT', 'MKRUSDT', 'DYDXUSDT'
]

# ====================== TRADE CONTROLS ======================
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "6"))
TRADE_COOLDOWN_MINUTES = int(os.getenv("TRADE_COOLDOWN_MINUTES", "45"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))

# ====================== RISK MANAGEMENT ======================
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.5"))  # 0.5% of equity per trade
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "2.0"))  # -2% → pause 24h
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "8.0"))  # -8% from peak → kill-switch
MAX_CORRELATED_EXPOSURE_PCT = float(os.getenv("MAX_CORRELATED_EXPOSURE_PCT", "30.0"))  # portfolio heat cap
ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER", "1.6"))
TP1_RR_RATIO = float(os.getenv("TP1_RR_RATIO", "2.0"))  # Risk:Reward for TP1
TP2_RR_RATIO = float(os.getenv("TP2_RR_RATIO", "3.5"))  # Risk:Reward for TP2

# ====================== STRATEGY ======================
MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "70"))
ML_MIN_PROBABILITY = float(os.getenv("ML_MIN_PROBABILITY", "0.65"))  # ML must agree with >65%
ADX_MIN_THRESHOLD = float(os.getenv("ADX_MIN_THRESHOLD", "25.0"))
VOLUME_CONFIRMATION_RATIO = float(os.getenv("VOLUME_CONFIRMATION_RATIO", "0.8"))
ENABLE_TRAILING_STOP = os.getenv("ENABLE_TRAILING_STOP", "YES").upper() == "YES"
ENABLE_ML_LAYER = os.getenv("ENABLE_ML_LAYER", "YES").upper() == "YES"

# ====================== ML CONFIGURATION ======================
ML_RETRAIN_HOUR = int(os.getenv("ML_RETRAIN_HOUR", "2"))  # Retrain at 2 AM UTC
ML_LOOKBACK_DAYS = int(os.getenv("ML_LOOKBACK_DAYS", "180"))  # 180 days training window
ML_WALK_FORWARD_TRAIN = int(os.getenv("ML_WALK_FORWARD_TRAIN", "90"))  # 90-day train window
ML_WALK_FORWARD_TEST = int(os.getenv("ML_WALK_FORWARD_TEST", "30"))  # 30-day test window

# ====================== BACKTESTER ======================
BACKTEST_MIN_YEARS = float(os.getenv("BACKTEST_MIN_YEARS", "0.5"))  # Minimum data for backtest
SLIPPAGE_TICKS_MIN = float(os.getenv("SLIPPAGE_TICKS_MIN", "0.5"))
SLIPPAGE_TICKS_MAX = float(os.getenv("SLIPPAGE_TICKS_MAX", "2.0"))
MAKER_FEE_PCT = float(os.getenv("MAKER_FEE_PCT", "0.02"))  # 0.02%
TAKER_FEE_PCT = float(os.getenv("TAKER_FEE_PCT", "0.04"))  # 0.04%
MONTE_CARLO_RUNS = int(os.getenv("MONTE_CARLO_RUNS", "10000"))

# ====================== EMAIL ======================
EMAIL_FROM = os.getenv("EMAIL_FROM", "")
EMAIL_TO = os.getenv("EMAIL_TO", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))

# ====================== MONITORING ======================
SCAN_INTERVAL_MINUTES = 15
HEALTH_CHECK_PORT = int(os.getenv("HEALTH_CHECK_PORT", "8080"))
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5000"))
PERFORMANCE_DEVIATION_THRESHOLD = float(os.getenv("PERFORMANCE_DEVIATION_THRESHOLD", "15.0"))

# ====================== DATA PATHS ======================
LOG_FILE = os.getenv("LOG_FILE", "signals_log.csv")
ML_MODEL_DIR = os.getenv("ML_MODEL_DIR", "ml_models")
HISTORICAL_DATA_DIR = os.getenv("HISTORICAL_DATA_DIR", "historical_data")
TRADE_JOURNAL_FILE = os.getenv("TRADE_JOURNAL_FILE", "trade_journal.csv")

# ====================== HIGH IMPACT NEWS DATES ======================
# Updated via economic calendar; add dates as YYYY-MM-DD
HIGH_IMPACT_DATES_STR = os.getenv("HIGH_IMPACT_DATES", "")
HIGH_IMPACT_DATES = [d.strip() for d in HIGH_IMPACT_DATES_STR.split(",") if d.strip()]

# ====================== CORRELATION GROUPS ======================
# Assets within same group are considered highly correlated
CORRELATION_GROUPS = {
    'BTC_GROUP': ['BTCUSDT', 'BCHUSDT', 'LTCUSDT'],
    'ETH_GROUP': ['ETHUSDT', 'OPUSDT'],
    'DEFI_GROUP': ['AAVEUSDT', 'SNXUSDT', 'CRVUSDT', 'MKRUSDT', 'DYDXUSDT'],
    'L1_GROUP': ['SOLUSDT', 'AVAXUSDT', 'DOTUSDT', 'ATOMUSDT', 'NEARUSDT', 'ICPUSDT', 'SEIUSDT', 'INJUSDT'],
    'MEME_GROUP': ['DOGEUSDT'],
}

# ====================== CIRCUIT BREAKER ======================
CIRCUIT_BREAKER_ATR_MULTIPLE = float(os.getenv("CIRCUIT_BREAKER_ATR_MULTIPLE", "3.0"))
CIRCUIT_BREAKER_LOOKBACK = int(os.getenv("CIRCUIT_BREAKER_LOOKBACK", "20"))  # 20-day ATR

# ====================== REGIME LABELS ======================
REGIME_TRENDING = 0
REGIME_RANGING = 1
REGIME_HIGH_VOLATILITY = 2
REGIME_LOW_LIQUIDITY = 3
REGIME_NAMES = {
    REGIME_TRENDING: "TRENDING",
    REGIME_RANGING: "RANGING",
    REGIME_HIGH_VOLATILITY: "HIGH_VOLATILITY",
    REGIME_LOW_LIQUIDITY: "LOW_LIQUIDITY",
}
