import datetime
import zoneinfo
from collections import deque
from dataclasses import dataclass
import numpy as np
import pandas as pd

_NY_TZ = zoneinfo.ZoneInfo("America/New_York")

# --- Tunable thresholds ------------------------------------------------
ADX_PERIOD = 14
ATR_PERIOD = 14
TREND_ADX_THRESHOLD = 25         # ADX at/above this => trend strength present
RANGE_ADX_THRESHOLD = 18         # ADX at/below this => weak/no trend
VWAP_STRETCH_TREND_ATR = 1.5     # |close - VWAP| in ATR units => confirms trend
RANGE_CONTAINMENT_MIN = 0.70     # fraction of recent closes inside OR => ranging

CONFIRMATION_VOTES = 3           # majority votes required
VOTE_HISTORY_LEN = 5             # trailing 5-min cycles window


@dataclass
class RegimeMetrics:
    adx: float
    atr: float
    atr_pct: float                 # ATR as % of price
    dist_from_vwap_atr: float      # signed: (close - vwap) / atr
    range_containment: float       # fraction of recent closes inside today's OR


@dataclass
class RegimeState:
    label: str          # "TRENDING" | "RANGING" | "TRANSITIONAL"
    confirmed: bool      # whether hysteresis has locked in this label this cycle
    raw_label: str        # this cycle's unconfirmed read, before hysteresis
    metrics: RegimeMetrics


_regime_votes: dict[str, deque] = {}
_confirmed_regime: dict[str, str] = {}
_last_session_date: dict[str, datetime.date] = {}


def _reset_if_new_session(anchor: str) -> None:
    today = datetime.datetime.now(_NY_TZ).date()
    if _last_session_date.get(anchor) != today:
        _regime_votes.pop(anchor, None)
        _confirmed_regime.pop(anchor, None)
        _last_session_date[anchor] = today


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures df.index is a clean, single-level DatetimeIndex."""
    if df is None or df.empty:
        return df

    df = df.copy()

    # Handle MultiIndex returned by Alpaca (symbol, timestamp)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    # Convert index to DatetimeIndex cleanly using utc=True for tz-aware objects
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index, utc=True)

    return df


# --- Indicator math --------------------------------------------------

def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    gap_tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # df.index is guaranteed to be a DatetimeIndex via _sanitize_dataframe.
    # .date works natively on DatetimeIndex without needing .dt or pd.to_datetime.
    dates_series = pd.Series(df.index.date, index=df.index)
    is_session_start = dates_series != dates_series.shift(1)

    no_gap_tr = df["high"] - df["low"]
    return gap_tr.where(~is_session_start, no_gap_tr)


def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    return _true_range(df).rolling(period, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = _true_range(df).rolling(period, min_periods=period).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period, min_periods=period).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period, min_periods=period).mean()


def _session_vwap(df_session: pd.DataFrame) -> pd.Series:
    typical_price = (df_session["high"] + df_session["low"] + df_session["close"]) / 3
    cum_vol = df_session["volume"].cumsum()
    cum_pv = (typical_price * df_session["volume"]).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


GAP_ATR_TREND_THRESHOLD = 0.5          
OPENING_BAR_RANGE_ATR_THRESHOLD = 0.6  


@dataclass
class EarlyWindowSignal:
    gap_atr: float
    opening_bar_range_atr: float
    candidate_trending: bool


def compute_early_window_signal(
    df_extended: pd.DataFrame,
    session_start: pd.Timestamp,
) -> EarlyWindowSignal | None:
    df_extended = _sanitize_dataframe(df_extended)
    if df_extended is None or df_extended.empty:
        return None

    # Align session_start timezone with index
    if df_extended.index.tz is not None and session_start.tzinfo is None:
        session_start = session_start.tz_localize(df_extended.index.tz)

    prior_bars = df_extended[df_extended.index < session_start]
    df_session = df_extended[df_extended.index >= session_start]

    if len(prior_bars) < ATR_PERIOD + 1 or df_session.empty:
        return None

    atr_series = _atr(prior_bars)
    atr_at_open = float(atr_series.iloc[-1])
    if atr_at_open <= 0 or np.isnan(atr_at_open):
        return None

    prior_close = float(prior_bars["close"].iloc[-1])
    open_col = "open" if "open" in df_session.columns else "close"
    today_open = float(df_session[open_col].iloc[0])
    gap_atr = abs(today_open - prior_close) / atr_at_open

    first_bar = df_session.iloc[0]
    opening_bar_range_atr = float(first_bar["high"] - first_bar["low"]) / atr_at_open

    candidate = (gap_atr >= GAP_ATR_TREND_THRESHOLD) or (opening_bar_range_atr >= OPENING_BAR_RANGE_ATR_THRESHOLD)

    return EarlyWindowSignal(
        gap_atr=gap_atr,
        opening_bar_range_atr=opening_bar_range_atr,
        candidate_trending=candidate,
    )


# --- Metrics + classification -------------------------------------------

def compute_regime_metrics(
    df_extended: pd.DataFrame,
    session_start: pd.Timestamp,
    opening_range_high: float,
    opening_range_low: float,
) -> RegimeMetrics | None:
    df_extended = _sanitize_dataframe(df_extended)

    required_bars = max(ADX_PERIOD, ATR_PERIOD) + 1
    if df_extended is None or len(df_extended) < required_bars:
        return None

    for col in ("high", "low", "close", "volume"):
        if col not in df_extended.columns:
            return None

    # Align session_start timezone with index
    if df_extended.index.tz is not None and session_start.tzinfo is None:
        session_start = session_start.tz_localize(df_extended.index.tz)

    df_session = df_extended[df_extended.index >= session_start]
    if df_session.empty:
        return None

    atr_series = _atr(df_extended)
    latest_atr = float(atr_series.iloc[-1])
    latest_close = float(df_extended["close"].iloc[-1])
    if latest_atr <= 0 or np.isnan(latest_atr):
        return None

    if len(df_session) < ADX_PERIOD + 1:
        return None
    adx_series = _adx(df_session)
    latest_adx = float(adx_series.iloc[-1])
    if np.isnan(latest_adx):
        return None

    vwap_series = _session_vwap(df_session)
    latest_vwap = float(vwap_series.iloc[-1])
    if np.isnan(latest_vwap):
        return None

    dist_from_vwap_atr = (latest_close - latest_vwap) / latest_atr

    or_width = opening_range_high - opening_range_low
    if or_width > 0:
        recent_closes = df_session["close"].tail(6)
        containment = float(
            ((recent_closes >= opening_range_low) & (recent_closes <= opening_range_high)).mean()
        )
    else:
        containment = 1.0

    return RegimeMetrics(
        adx=latest_adx,
        atr=latest_atr,
        atr_pct=latest_atr / latest_close,
        dist_from_vwap_atr=dist_from_vwap_atr,
        range_containment=containment,
    )


def _raw_classify(m: RegimeMetrics) -> str:
    if m.adx >= TREND_ADX_THRESHOLD and abs(m.dist_from_vwap_atr) >= VWAP_STRETCH_TREND_ATR:
        return "TRENDING"
    if m.adx <= RANGE_ADX_THRESHOLD and m.range_containment >= RANGE_CONTAINMENT_MIN:
        return "RANGING"
    return "TRANSITIONAL"


def resolve_regime(anchor: str, m: RegimeMetrics) -> RegimeState:
    _reset_if_new_session(anchor)

    raw = _raw_classify(m)

    votes = _regime_votes.setdefault(anchor, deque(maxlen=VOTE_HISTORY_LEN))
    votes.append(raw)

    counts = {label: list(votes).count(label) for label in ("TRENDING", "RANGING", "TRANSITIONAL")}
    majority_label, majority_count = max(counts.items(), key=lambda kv: kv[1])

    prior_confirmed = _confirmed_regime.get(anchor, "RANGING")
    confirmed = majority_count >= CONFIRMATION_VOTES

    if confirmed:
        _confirmed_regime[anchor] = majority_label
        active_label = majority_label
    else:
        active_label = prior_confirmed

    return RegimeState(label=active_label, confirmed=confirmed, raw_label=raw, metrics=m)