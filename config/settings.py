import os

# =====================================================================
# CONFIGURATION & SECURITY
# =====================================================================

ALPACA_API_KEY = ""
ALPACA_SECRET_KEY = ""
GEMINI_API_KEY = ""

# TELEGRAM CREDENTIALS
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# CURRENTS API CREDENTIALS
CURRENTS_API_KEY = ""

# LOGGING
LOG_FILE_PATH = "" #Where trade log JSON file is located. If no file, program will create by itself.


#Risk management variables
RISK_DOLLARS_PER_TRADE = 200.0      # $ risked per trade, sized off (entry - stop) distance
MAX_NOTIONAL_PER_TRADE = 10000.0     # usually 500 -- made it higher for testing hard $ cap on (qty * entry_price), independent of the
                                     # above -- backstops position size when the stop distance
                                     # ends up small (tight LLM stop, or narrow leveraged OR)
MIN_TP1_RR = 1.00                   # sweep strategy: TP1 must be at least this many R away
MIN_SL_DISTANCE_PCT = 0.80           # sweep strategy: floor on how far Gemini's SL may sit
                                       # below the system's own scaled_risk_pct baseline

# DYNAMIC STRATEGY ROUTING TABLE 
WATCHLIST_MATRIX = { #STOCK ETF TICKERS
    "NASDAQ_LEVERAGED": {
        "anchor": "QQQ",
        "long_target": "TQQQ",
        "short_target": "SQQQ",
        "leverage_multiplier": 3.0
    },
    "SP500_LEVERAGED": {
        "anchor": "SPY",
        "long_target": "SPXL",
        "short_target": "SPXS",
        "leverage_multiplier": 3.0
    },
    "SMALLCAP_LEVERAGED": {
        "anchor": "IWM",
        "long_target": "TNA",
        "short_target": "TZA",
        "leverage_multiplier": 3.0
    },
    "SEMIS_LEVERAGED": {
        "anchor": "SOXX",
        "long_target": "SOXL",
        "short_target": "SOXS",
        "leverage_multiplier": 3.0
    },
    "FINANCIALS_LEVERAGED": {
        "anchor": "XLF",
        "long_target": "FAS",
        "short_target": "FAZ",
        "leverage_multiplier": 3.0
    }
}

COMMODITY_MATRIX = { #COMMODITY TICKERS
    "GOLD": {
        "anchor_etf": "GLD",          
        "futures_yf_ticker": "GC=F",
        "long_target": "GLD",           
        "long_leverage_multiplier": 1.0,
        "short_target": "GLL",          
        "short_leverage_multiplier": 2.0, 
    },
    "SILVER": {
        "anchor_etf": "SLV",         
        "futures_yf_ticker": "SI=F",   
        "long_target": "SLV",
        "long_leverage_multiplier": 1.0,
        "short_target": "ZSL",          
        "short_leverage_multiplier": 2.0,
    },
    "CRUDE_OIL": {
        "anchor_etf": "USO",         
        "futures_yf_ticker": "CL=F",    
        "long_target": "USO",
        "long_leverage_multiplier": 1.0,
        "short_target": "SCO",          
        "short_leverage_multiplier": 2.0,
    },
    "NAT_GAS": {
        "anchor_etf": "UNG",         
        "futures_yf_ticker": "NG=F",   
        "long_target": "UNG",
        "long_leverage_multiplier": 1.0,
        "short_target": "KOLD",          
        "short_leverage_multiplier": 2.0,
    }
#    "AGRICULTURE": {
#        "anchor_etf": "DBA",         
#        "futures_yf_ticker": "DBA",   
#        "long_target": "DBA",
#        "long_leverage_multiplier": 1.0,
#        "short_target": "SMN",          
#        "short_leverage_multiplier": 2.0,
#    }
}

#     "COPPER": {
#        "anchor_etf": "CPER",         
#        "futures_yf_ticker": "HG=F",   
#        "long_target": "CPER",
#        "long_leverage_multiplier": 1.0,
#        "short_target": "NO SHORT ETF",          
#        "short_leverage_multiplier": 2.0,
#    },

ALL_EXECUTION_TICKERS = []
for cfg in WATCHLIST_MATRIX.values():
    ALL_EXECUTION_TICKERS.extend([cfg["long_target"], cfg["short_target"]])
