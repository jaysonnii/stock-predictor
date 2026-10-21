"""
Tickers used to train the pooled model. Stocks and ETFs across sectors.
Add or remove symbols freely. Symbols with too little history are skipped
automatically at training time.
"""

BENCHMARK = "SPY"  # excluded from training; the model predicts return relative to this

_STOCKS = [
    # Mega-cap tech and software
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    "ADBE", "CRM", "CSCO", "IBM", "INTC", "AMD", "QCOM", "TXN", "NFLX", "PYPL",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "BRK-B", "AXP", "SCHW",
    # Healthcare
    "JNJ", "UNH", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Consumer
    "HD", "LOW", "MCD", "SBUX", "NKE", "WMT", "COST", "TGT", "PG", "KO", "PEP", "CL",
    # Industrials
    "CAT", "DE", "BA", "HON", "GE", "LMT", "RTX", "UPS", "FDX", "UNP",
    # Communication and utilities
    "DIS", "CMCSA", "T", "VZ", "NEE", "DUK",
]

_ETFS = [
    "QQQ", "IWM", "DIA", "VTI", "VOO", "XLF", "XLE", "XLK", "XLV", "XLP",
    "XLU", "XLY", "XLI", "EFA", "EEM", "AGG", "TLT", "GLD", "VNQ", "SCHD",
    "DVY", "HDV", "VYM",
]

# dict.fromkeys removes duplicates and keeps order.
TICKERS = [t for t in dict.fromkeys(_STOCKS + _ETFS) if t != BENCHMARK]
