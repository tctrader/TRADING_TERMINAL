#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════
  FINCEPT INTELLIGENCE TRADER v2
  Swing + Positional | Market-Hours Aware

  Session logic:
  - US Market open (Mon–Fri 9:30–16:00 ET, ex-holidays)
    → trades stocks + ETFs + crypto
  - Pre/Post market (4:00–9:30 ET)
    → watches news, prepares signals, NO entries
  - Market closed (nights + weekends)
    → trades crypto only (24/7)
    → scans news for next-open watchlist
═══════════════════════════════════════════════════════
"""

import os, json, time, logging, requests, pytz
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
from gnews import GNews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ═══════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════
C = {
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_CHAT_IDS":  [x.strip() for x in os.getenv("TELEGRAM_CHAT_IDS", os.getenv("TELEGRAM_CHAT_ID","")).split(",") if x.strip()],
    "PAPER_MODE":         os.getenv("PAPER_MODE", "true").lower() != "false",
    "FINCEPT_API_KEY":    os.getenv("FINCEPT_API_KEY", ""),
    "ALPACA_API_KEY":     os.getenv("ALPACA_API_KEY", ""),
    "ALPACA_SECRET_KEY":  os.getenv("ALPACA_SECRET_KEY", ""),

    # Position sizing
    "POSITION_SIZE_USD":   float(os.getenv("POSITION_SIZE_USD",  "100")),
    "MAX_STOCK_POSITIONS": int(os.getenv("MAX_STOCK_POSITIONS",   "3")),
    "MAX_CRYPTO_POSITIONS":int(os.getenv("MAX_CRYPTO_POSITIONS",  "2")),
    "DAILY_LOSS_LIMIT":    float(os.getenv("DAILY_LOSS_LIMIT",   "150")),

    # Scan timing (seconds)
    "SCAN_INTERVAL_OPEN":    int(os.getenv("SCAN_INTERVAL_OPEN",   "300")),  # 5 min during market
    "SCAN_INTERVAL_PREPOST": int(os.getenv("SCAN_INTERVAL_PREPOST","600")),  # 10 min pre/post
    "SCAN_INTERVAL_CLOSED":  int(os.getenv("SCAN_INTERVAL_CLOSED","1800")),  # 30 min closed

    # Signal thresholds
    "MIN_SENTIMENT_SCORE":  float(os.getenv("MIN_SENTIMENT_SCORE", "0.25")),
    "MIN_COMPOSITE_SCORE":  float(os.getenv("MIN_COMPOSITE_SCORE", "58")),

    # SL/TP
    "SL_PCT":     float(os.getenv("SL_PCT",    "2.5")),
    "TP1_PCT":    float(os.getenv("TP1_PCT",   "4.0")),
    "TP2_PCT":    float(os.getenv("TP2_PCT",   "8.0")),
    "TP3_PCT":    float(os.getenv("TP3_PCT",  "15.0")),
    "TRAIL_PCT":  float(os.getenv("TRAIL_PCT",  "2.0")),

    # Positional hold — swing can hold overnight for stocks
    "MAX_HOLD_DAYS_STOCK":  int(os.getenv("MAX_HOLD_DAYS_STOCK",  "5")),
    "MAX_HOLD_DAYS_CRYPTO": int(os.getenv("MAX_HOLD_DAYS_CRYPTO", "3")),

    "STATE_FILE": "state.json",
}

# ═══════════════════════════════════════════════════════
#  MARKET CALENDAR
# ═══════════════════════════════════════════════════════

# US market holidays 2025-2026
US_HOLIDAYS = {
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
}

def get_market_session() -> dict:
    """
    Returns current market session info.
    Sessions: OPEN | PRE | POST | CLOSED | HOLIDAY | WEEKEND
    """
    now_et = datetime.now(ET)
    today  = now_et.date()
    wd     = now_et.weekday()  # 0=Mon, 6=Sun
    t      = now_et.time()

    from datetime import time as dtime
    PRE_OPEN  = dtime(4, 0)
    OPEN      = dtime(9, 30)
    CLOSE     = dtime(16, 0)
    POST_CLOSE= dtime(20, 0)

    if wd >= 5:  # Saturday or Sunday
        return {
            "session":    "WEEKEND",
            "can_trade_stocks": False,
            "can_trade_crypto": True,
            "scan_interval":    C["SCAN_INTERVAL_CLOSED"],
            "label":      f"Weekend — Market opens Monday 9:30 ET",
            "next_open":  _next_open(now_et),
        }

    if today in US_HOLIDAYS:
        return {
            "session":    "HOLIDAY",
            "can_trade_stocks": False,
            "can_trade_crypto": True,
            "scan_interval":    C["SCAN_INTERVAL_CLOSED"],
            "label":      "US Market Holiday — Crypto only",
            "next_open":  _next_open(now_et),
        }

    if t < PRE_OPEN:
        return {
            "session":    "CLOSED",
            "can_trade_stocks": False,
            "can_trade_crypto": True,
            "scan_interval":    C["SCAN_INTERVAL_CLOSED"],
            "label":      f"Market closed — opens at 9:30 ET",
            "next_open":  now_et.replace(hour=9, minute=30, second=0),
        }

    if PRE_OPEN <= t < OPEN:
        mins = int((datetime.combine(today, OPEN) - datetime.combine(today, t)).seconds / 60)
        return {
            "session":    "PRE",
            "can_trade_stocks": False,  # no entries pre-market
            "can_trade_crypto": True,
            "scan_interval":    C["SCAN_INTERVAL_PREPOST"],
            "label":      f"Pre-market — {mins}min until open",
            "next_open":  now_et.replace(hour=9, minute=30, second=0),
        }

    if OPEN <= t < CLOSE:
        mins_left = int((datetime.combine(today, CLOSE) - datetime.combine(today, t)).seconds / 60)
        return {
            "session":    "OPEN",
            "can_trade_stocks": True,
            "can_trade_crypto": True,
            "scan_interval":    C["SCAN_INTERVAL_OPEN"],
            "label":      f"Market OPEN — {mins_left}min remaining",
            "next_open":  None,
        }

    if CLOSE <= t < POST_CLOSE:
        return {
            "session":    "POST",
            "can_trade_stocks": False,
            "can_trade_crypto": True,
            "scan_interval":    C["SCAN_INTERVAL_PREPOST"],
            "label":      "After-hours — No new stock entries",
            "next_open":  _next_open(now_et),
        }

    return {
        "session":    "CLOSED",
        "can_trade_stocks": False,
        "can_trade_crypto": True,
        "scan_interval":    C["SCAN_INTERVAL_CLOSED"],
        "label":      "Market closed",
        "next_open":  _next_open(now_et),
    }

def _next_open(now_et: datetime) -> datetime:
    """Find next market open (skipping weekends + holidays)"""
    candidate = now_et + timedelta(days=1)
    candidate = candidate.replace(hour=9, minute=30, second=0, microsecond=0)
    for _ in range(10):
        if candidate.weekday() < 5 and candidate.date() not in US_HOLIDAYS:
            return candidate
        candidate += timedelta(days=1)
    return candidate

def fmt_next_open(session: dict) -> str:
    no = session.get("next_open")
    if not no:
        return ""
    now = datetime.now(ET)
    delta = no.replace(tzinfo=ET) - now if no.tzinfo is None else no - now
    h = int(delta.total_seconds() // 3600)
    m = int((delta.total_seconds() % 3600) // 60)
    return f"{no.strftime('%a %b %d at 9:30 ET')} ({h}h {m}m away)"

# ═══════════════════════════════════════════════════════
#  WATCHLISTS — split by type
# ═══════════════════════════════════════════════════════
STOCK_WATCHLIST = {
    "AAPL":  {"name": "Apple",      "sector": "tech"},
    "NVDA":  {"name": "NVIDIA",     "sector": "ai"},
    "TSLA":  {"name": "Tesla",      "sector": "ev"},
    "MSFT":  {"name": "Microsoft",  "sector": "tech"},
    "AMZN":  {"name": "Amazon",     "sector": "retail"},
    "META":  {"name": "Meta",       "sector": "social"},
    "GOOGL": {"name": "Google",     "sector": "tech"},
    "SPY":   {"name": "S&P 500 ETF","sector": "macro"},
    "QQQ":   {"name": "Nasdaq ETF", "sector": "macro"},
    "GLD":   {"name": "Gold ETF",   "sector": "safe"},
    "TLT":   {"name": "Bonds ETF",  "sector": "rates"},
}

CRYPTO_WATCHLIST = {
    "BTC-USD": {"name": "Bitcoin",  "sector": "crypto"},
    "ETH-USD": {"name": "Ethereum", "sector": "crypto"},
    "SOL-USD": {"name": "Solana",   "sector": "crypto"},
    "BNB-USD": {"name": "BNB",      "sector": "crypto"},
}

NEWS_TRIGGERS = {
    "federal reserve": ["TLT","GLD","SPY","QQQ"],
    "interest rate":   ["TLT","GLD","SPY","QQQ"],
    "inflation":       ["GLD","TLT","SPY"],
    "cpi":             ["GLD","TLT","SPY"],
    "powell":          ["SPY","QQQ","TLT"],
    "recession":       ["GLD","TLT","SPY"],
    "ai":              ["NVDA","MSFT","META","AAPL"],
    "artificial intelligence": ["NVDA","MSFT"],
    "chip":            ["NVDA","AAPL"],
    "earnings":        ["AAPL","MSFT","NVDA","META","AMZN","GOOGL"],
    "tesla":           ["TSLA"],
    "musk":            ["TSLA","BTC-USD"],
    "bitcoin":         ["BTC-USD","ETH-USD"],
    "crypto":          ["BTC-USD","ETH-USD","SOL-USD"],
    "ethereum":        ["ETH-USD"],
    "sec":             ["BTC-USD","ETH-USD"],
    "etf":             ["BTC-USD","SPY","QQQ"],
    "war":             ["GLD","TLT"],
    "geopolitical":    ["GLD","TLT","SPY"],
    "gdp":             ["SPY","QQQ"],
    "jobs":            ["SPY","TLT"],
    "amazon":          ["AMZN"],
    "google":          ["GOOGL"],
    "meta":            ["META"],
    "microsoft":       ["MSFT"],
    "apple":           ["AAPL"],
    "nvidia":          ["NVDA"],
}

# ═══════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════
STATE = {
    "balance":      C["POSITION_SIZE_USD"] * 10,
    "positions":    {},
    "pending_signals": {},   # built during pre-market for next open
    "trade_log":    [],
    "scan_count":   0,
    "total_pnl":    0.0,
    "daily_pnl":    0.0,
    "daily_reset":  datetime.now().strftime("%Y-%m-%d"),
    "wins": 0, "losses": 0,
    "macro_context": {},
    "last_session":  "",
    "started_at":   time.time(),
}

def load_state():
    try:
        if Path(C["STATE_FILE"]).exists():
            saved = json.loads(Path(C["STATE_FILE"]).read_text())
            STATE.update(saved)
            log.info("✅ State loaded")
    except Exception as e:
        log.warning(f"State load: {e}")

def save_state():
    try:
        Path(C["STATE_FILE"]).write_text(json.dumps(STATE, indent=2, default=str))
    except Exception as e:
        log.warning(f"Save: {e}")

# ═══════════════════════════════════════════════════════
#  TELEGRAM
# ═══════════════════════════════════════════════════════
def tg(msg: str):
    if not C["TELEGRAM_BOT_TOKEN"] or not C["TELEGRAM_CHAT_IDS"]:
        log.info(f"📵 TG: {msg[:100]}")
        return
    for chat_id in C["TELEGRAM_CHAT_IDS"]:
        try:
            requests.post(
                f"https://api.telegram.org/bot{C['TELEGRAM_BOT_TOKEN']}/sendMessage",
                json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
                timeout=10
            )
        except Exception as e:
            log.warning(f"TG: {e}")

# ═══════════════════════════════════════════════════════
#  SENTIMENT ENGINE
# ═══════════════════════════════════════════════════════
vader = SentimentIntensityAnalyzer()

def score_news(items: list, symbol: str) -> dict:
    scores = []
    for item in items[:15]:
        text = f"{item.get('title','')} {item.get('description','') or ''}"
        if text.strip():
            s = vader.polarity_scores(text)["compound"]
            scores.append(s)
        title = (item.get("title","") or "").lower()
        if any(w in title for w in ["surge","soars","beats","record","rally","upgrade","partnership","bullish"]):
            scores.append(0.5)
        if any(w in title for w in ["crash","drops","miss","lawsuit","probe","ban","fraud","warning","downgrade","bearish"]):
            scores.append(-0.5)

    if not scores:
        return {"score": 0.0, "direction": "NEUTRAL", "count": 0, "confidence": 0}

    avg   = float(np.mean(scores))
    conf  = min(100, len(scores) * 10)
    direc = "BULLISH" if avg >= 0.2 else "BEARISH" if avg <= -0.2 else "NEUTRAL"
    return {"score": round(avg,3), "direction": direc, "count": len(items), "confidence": conf}

# ═══════════════════════════════════════════════════════
#  TECHNICALS
# ═══════════════════════════════════════════════════════
def get_technicals(symbol: str, interval: str = "1h") -> dict | None:
    try:
        period = "10d" if interval == "1h" else "60d"
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty or len(df) < 20:
            return None

        close = df["Close"]
        vol   = df["Volume"]
        high  = df["High"]
        low   = df["Low"]
        price = float(close.iloc[-1])

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = float((100 - 100/(1+rs)).iloc[-1])

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9).mean()
        macd_hist = float((macd - sig).iloc[-1])
        macd_dir  = "BULLISH" if macd.iloc[-1] > sig.iloc[-1] else "BEARISH"
        macd_cross= "CROSS_UP"   if macd.iloc[-1] > sig.iloc[-1] and macd.iloc[-2] <= sig.iloc[-2] else \
                    "CROSS_DOWN" if macd.iloc[-1] < sig.iloc[-1] and macd.iloc[-2] >= sig.iloc[-2] else macd_dir

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_lo = float((sma20 - 2*std20).iloc[-1])
        bb_up = float((sma20 + 2*std20).iloc[-1])
        bb_pos= (price - bb_lo) / max(bb_up - bb_lo, 0.0001)

        # Volume
        avg_vol   = float(vol.rolling(20).mean().iloc[-1])
        vol_ratio = float(vol.iloc[-1]) / avg_vol if avg_vol > 0 else 1

        # ATR
        tr    = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr   = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = (atr/price)*100

        # Price changes
        ch1  = ((price - float(close.iloc[-2]))/float(close.iloc[-2]))*100 if len(close)>1 else 0
        ch24 = ((price - float(close.iloc[-25]))/float(close.iloc[-25]))*100 if len(close)>24 else ch1*24
        ch5d = ((price - float(close.iloc[-5*24]))/float(close.iloc[-5*24]))*100 if len(close)>5*24 else ch24

        # Support / Resistance (20-period high/low)
        support    = float(low.rolling(20).min().iloc[-1])
        resistance = float(high.rolling(20).max().iloc[-1])
        dist_to_res= ((resistance-price)/price)*100
        dist_to_sup= ((price-support)/price)*100

        # Score
        score = 50.0
        if rsi < 30:    score += 22
        elif rsi < 45:  score += 12
        elif rsi > 75:  score -= 22
        elif rsi > 60:  score -= 10
        if "CROSS_UP"   in macd_cross: score += 15
        if "CROSS_DOWN" in macd_cross: score -= 15
        elif macd_dir == "BULLISH":    score += 7
        elif macd_dir == "BEARISH":    score -= 7
        if bb_pos < 0.15: score += 12  # near lower band
        if bb_pos > 0.85: score -= 12
        if vol_ratio > 2:  score += 10
        if vol_ratio > 3:  score += 5
        if ch1  > 0.5:  score += 6
        if ch1  < -0.5: score -= 6
        if ch24 > 2:    score += 5
        if ch24 < -2:   score -= 5
        # Near support = good entry for swing
        if dist_to_sup < 2:  score += 8

        score = max(0, min(100, score))

        return {
            "price": round(price,4), "rsi": round(rsi,1),
            "macd": macd_cross, "macd_hist": round(macd_hist,4),
            "bb_pos": round(bb_pos,2), "bb_lo": round(bb_lo,4), "bb_up": round(bb_up,4),
            "vol_ratio": round(vol_ratio,1),
            "ch1h": round(ch1,2), "ch24h": round(ch24,2), "ch5d": round(ch5d,2),
            "atr_pct": round(atr_pct,2),
            "support": round(support,4), "resistance": round(resistance,4),
            "dist_to_res": round(dist_to_res,2),
            "score": round(score,1),
        }
    except Exception as e:
        log.warning(f"Tech {symbol}: {e}")
        return None

# ═══════════════════════════════════════════════════════
#  MACRO
# ═══════════════════════════════════════════════════════
def fetch_macro() -> dict:
    macro = {}
    try:
        vix = yf.Ticker("^VIX").history(period="2d")
        if not vix.empty:
            macro["vix"]  = round(float(vix["Close"].iloc[-1]),1)
            macro["vix_ch"]= round(((float(vix["Close"].iloc[-1])-float(vix["Close"].iloc[-2]))/float(vix["Close"].iloc[-2]))*100,2)

        dxy = yf.Ticker("DX-Y.NYB").history(period="2d")
        if not dxy.empty:
            macro["dxy"]  = round(float(dxy["Close"].iloc[-1]),2)
            macro["dxy_ch"]= round(((float(dxy["Close"].iloc[-1])-float(dxy["Close"].iloc[-2]))/float(dxy["Close"].iloc[-2]))*100,2)

        tnx = yf.Ticker("^TNX").history(period="2d")
        if not tnx.empty:
            macro["yield_10y"] = round(float(tnx["Close"].iloc[-1]),2)

        btc = yf.Ticker("BTC-USD").history(period="2d")
        if not btc.empty:
            macro["btc_24h"] = round(((float(btc["Close"].iloc[-1])-float(btc["Close"].iloc[-2]))/float(btc["Close"].iloc[-2]))*100,2)

        vix_val = macro.get("vix", 20)
        if vix_val > 28:
            macro["regime"] = "RISK_OFF"
            macro["stock_bias"] = "DEFENSIVE"
            macro["favors"]     = ["GLD","TLT"]
            macro["avoids"]     = ["TSLA","NVDA","BTC-USD"]
        elif vix_val < 16:
            macro["regime"] = "RISK_ON"
            macro["stock_bias"] = "GROWTH"
            macro["favors"]     = ["NVDA","TSLA","QQQ","META","BTC-USD","ETH-USD"]
            macro["avoids"]     = ["TLT","GLD"]
        else:
            macro["regime"] = "NEUTRAL"
            macro["stock_bias"] = "BALANCED"
            macro["favors"]     = ["AAPL","MSFT","SPY","ETH-USD"]
            macro["avoids"]     = []

    except Exception as e:
        log.warning(f"Macro: {e}")
        macro.setdefault("regime","NEUTRAL")
        macro.setdefault("favors",["SPY"])
        macro.setdefault("avoids",[])

    STATE["macro_context"] = macro
    return macro

# ═══════════════════════════════════════════════════════
#  SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════
def generate_signal(symbol: str, news: list, macro: dict, asset_type: str = "stock") -> dict:
    sentiment = score_news(news, symbol)
    interval  = "1h" if asset_type == "stock" else "1h"
    tech      = get_technicals(symbol, interval)
    if not tech:
        return {"action":"HOLD","score":0,"reason":["No price data"]}

    price = tech["price"]

    # Macro alignment
    favors = macro.get("favors",[])
    avoids = macro.get("avoids",[])
    macro_score = 60
    if symbol in favors: macro_score = 75
    if symbol in avoids: macro_score = 35

    # Composite (sentiment 35% | tech 50% | macro 15%)
    sent_norm = (sentiment["score"]+1)/2*100
    composite = round(sent_norm*0.35 + tech["score"]*0.50 + macro_score*0.15, 1)

    # ATR-adjusted SL/TP
    atr      = tech["atr_pct"]
    sl_pct   = max(C["SL_PCT"],  atr*1.5)
    tp1_pct  = max(C["TP1_PCT"], atr*2.0)
    tp2_pct  = max(C["TP2_PCT"], atr*4.0)
    tp3_pct  = max(C["TP3_PCT"], atr*7.0)

    # Use support as SL anchor for swing trades
    if asset_type == "stock" and tech.get("support"):
        support_sl = ((price - tech["support"])/price)*100
        sl_pct = min(sl_pct, support_sl + 0.5)  # tight to support

    sl  = round(price*(1-sl_pct/100), 4)
    tp1 = round(price*(1+tp1_pct/100), 4)
    tp2 = round(price*(1+tp2_pct/100), 4)
    tp3 = round(price*(1+tp3_pct/100), 4)

    reasons = [
        f"News: {sentiment['direction']} ({sentiment['score']:+.2f}, {sentiment['count']} articles)",
        f"RSI: {tech['rsi']} | MACD: {tech['macd']} | Vol: {tech['vol_ratio']}×",
        f"BB pos: {tech['bb_pos']} | 1h: {tech['ch1h']:+.2f}% | 5d: {tech['ch5d']:+.2f}%",
        f"Macro: {macro.get('regime')} | VIX: {macro.get('vix','?')}",
        f"Support: ${tech.get('support','?')} | Resistance: ${tech.get('resistance','?')}",
    ]

    # Decision
    action = "HOLD"
    if (composite >= C["MIN_COMPOSITE_SCORE"] and
        sentiment["score"] >= C["MIN_SENTIMENT_SCORE"] and
        tech["rsi"] < 72):
        action = "BUY"
    elif (composite <= (100-C["MIN_COMPOSITE_SCORE"]) and
          sentiment["score"] <= -C["MIN_SENTIMENT_SCORE"]):
        action = "BEARISH"  # informational

    # Extra: don't buy into resistance
    if action == "BUY" and tech.get("dist_to_res",100) < 1.5:
        action = "HOLD"
        reasons.append("⚠️ Too close to resistance — waiting for breakout")

    return {
        "action": action, "score": composite, "price": price,
        "sl": sl, "sl_pct": round(sl_pct,2),
        "tp1": tp1, "tp1_pct": round(tp1_pct,2),
        "tp2": tp2, "tp2_pct": round(tp2_pct,2),
        "tp3": tp3, "tp3_pct": round(tp3_pct,2),
        "sentiment": sentiment, "tech": tech,
        "reasons": reasons, "asset_type": asset_type,
        "time": datetime.now().isoformat(),
    }

# ═══════════════════════════════════════════════════════
#  POSITION MANAGER
# ═══════════════════════════════════════════════════════
def check_exits(session: dict):
    for sym, pos in list(STATE["positions"].items()):
        try:
            asset_type = pos.get("asset_type","stock")
            interval   = "5m" if session["session"] == "OPEN" else "1h"
            data       = yf.Ticker(sym).history(period="1d", interval=interval)
            if data.empty:
                continue

            price  = float(data["Close"].iloc[-1])
            entry  = pos["entry_price"]
            pct    = ((price-entry)/entry)*100
            peak   = max(pos.get("peak",entry), price)
            trail  = peak*(1-C["TRAIL_PCT"]/100)
            held_d = (time.time()-pos["entry_ts"])/86400
            max_d  = C["MAX_HOLD_DAYS_STOCK"] if asset_type=="stock" else C["MAX_HOLD_DAYS_CRYPTO"]

            STATE["positions"][sym]["peak"]        = peak
            STATE["positions"][sym]["current_pct"] = round(pct,2)
            STATE["positions"][sym]["current_price"]= round(price,4)

            exit_reason = None

            # Hard stop
            if price <= pos["sl"]:
                exit_reason = f"🛑 SL hit ${price:.4f} ({pct:+.2f}%)"

            # TP ladder
            elif price >= pos["tp3"]:
                exit_reason = f"🏆 TP3 hit ${price:.4f} ({pct:+.2f}%)"

            elif price >= pos["tp2"] and not pos.get("tp2_hit"):
                # Partial: sell 50%
                half    = pos["size"]*0.5
                half_pnl= half*(pct/100)
                STATE["balance"]   += half+half_pnl
                STATE["total_pnl"] += half_pnl
                STATE["daily_pnl"] += half_pnl
                STATE["positions"][sym]["size"]   -= half
                STATE["positions"][sym]["tp2_hit"] = True
                tg(f"🎯 <b>TP2 PARTIAL — {sym}</b>\n"
                   f"Sold 50% at ${price:.4f} (+{pct:.2f}%)\n"
                   f"Profit: +${half_pnl:.2f} | Trailing rest at {C['TRAIL_PCT']}%")
                continue

            elif price >= pos["tp1"] and not pos.get("tp1_hit"):
                # Move SL to breakeven
                STATE["positions"][sym]["sl"]      = round(entry*1.002,4)
                STATE["positions"][sym]["tp1_hit"] = True
                tg(f"✅ <b>TP1 HIT — {sym}</b>\n"
                   f"${price:.4f} (+{pct:.2f}%) — SL moved to breakeven")
                continue

            # Trailing stop (only after TP1)
            elif price <= trail and pos.get("tp1_hit"):
                exit_reason = f"📉 Trail stop ({C['TRAIL_PCT']}% from peak {((peak-entry)/entry*100):+.2f}%)"

            # Max hold (swing/positional limit)
            elif held_d >= max_d:
                exit_reason = f"⏰ Max hold {max_d}d reached at {pct:+.2f}%"

            # Stock: force close before market closes (30 min buffer)
            # — Only if not a positional hold (user can configure)
            # Currently we allow overnight holds for swing

            if exit_reason:
                pnl = pos["size"]*(pct/100)
                STATE["balance"]   += pos["size"]+pnl
                STATE["total_pnl"] += pnl
                STATE["daily_pnl"] += pnl
                if pnl>=0: STATE["wins"]+=1
                else:      STATE["losses"]+=1

                STATE["trade_log"].insert(0,{
                    "type":"EXIT","symbol":sym,
                    "entry":entry,"exit":price,
                    "pnl":round(pnl,2),"pct":round(pct,2),
                    "held_days":round(held_d,2),
                    "reason":exit_reason,
                    "time":datetime.now().isoformat(),
                })

                wr = round(STATE["wins"]/(STATE["wins"]+STATE["losses"])*100) if STATE["wins"]+STATE["losses"]>0 else 0
                tg(f"{'🟢' if pnl>=0 else '🔴'} <b>EXIT — {sym}</b> {'📝' if C['PAPER_MODE'] else '💰'}\n\n"
                   f"💵 P&L: {'+' if pnl>=0 else ''}${pnl:.2f} ({pct:+.2f}%)\n"
                   f"⏱ Held: {held_d:.1f} days\n"
                   f"📌 {exit_reason}\n\n"
                   f"💰 Balance: ${STATE['balance']:.2f}\n"
                   f"📊 WR: {wr}% | Total P&L: {'+' if STATE['total_pnl']>=0 else ''}${STATE['total_pnl']:.2f}")

                del STATE["positions"][sym]
                log.info(f"EXIT {sym} {pct:+.2f}% ${pnl:.2f} | {exit_reason}")

        except Exception as e:
            log.warning(f"Exit {sym}: {e}")

def enter_position(sym: str, signal: dict, asset_type: str):
    if STATE["daily_pnl"] <= -C["DAILY_LOSS_LIMIT"]:
        return
    if sym in STATE["positions"]:
        return

    max_pos = C["MAX_STOCK_POSITIONS"] if asset_type=="stock" else C["MAX_CRYPTO_POSITIONS"]
    current = sum(1 for p in STATE["positions"].values() if p.get("asset_type")==asset_type)
    if current >= max_pos:
        return
    if STATE["balance"] < C["POSITION_SIZE_USD"]:
        return

    size  = C["POSITION_SIZE_USD"]
    price = signal["price"]
    STATE["balance"] -= size

    STATE["positions"][sym] = {
        "symbol":sym,"entry_price":price,"size":size,
        "sl":signal["sl"],"tp1":signal["tp1"],"tp2":signal["tp2"],"tp3":signal["tp3"],
        "sl_pct":signal["sl_pct"],"tp1_pct":signal["tp1_pct"],
        "peak":price,"current_pct":0,"current_price":price,
        "score":signal["score"],"asset_type":asset_type,
        "entry_ts":time.time(),"tp1_hit":False,"tp2_hit":False,
    }
    STATE["trade_log"].insert(0,{
        "type":"ENTRY","symbol":sym,"price":price,
        "size":size,"score":signal["score"],
        "asset_type":asset_type,"time":datetime.now().isoformat(),
    })

    t    = signal["tech"]
    s    = signal["sentiment"]
    r    = "\n  ".join(signal["reasons"])
    days = C["MAX_HOLD_DAYS_STOCK"] if asset_type=="stock" else C["MAX_HOLD_DAYS_CRYPTO"]

    tg(f"⚡ <b>{'SWING' if asset_type=='stock' else 'CRYPTO'} ENTRY — {sym}</b> {'📝' if C['PAPER_MODE'] else '💰'}\n\n"
       f"🎯 Score: {signal['score']}/100\n"
       f"💲 Price: ${price} | 💵 ${size:.0f}\n"
       f"Max hold: {days} days\n\n"
       f"🛑 SL:  ${signal['sl']} (-{signal['sl_pct']}%)\n"
       f"✅ TP1: ${signal['tp1']} (+{signal['tp1_pct']}%)\n"
       f"🎯 TP2: ${signal['tp2']} (+{signal['tp2_pct']}%)\n"
       f"🏆 TP3: ${signal['tp3']} (+{signal['tp3_pct']}%)\n\n"
       f"📰 {s['direction']} ({s['score']:+.2f}) | {s['count']} articles\n"
       f"📊 RSI:{t['rsi']} MACD:{t['macd']} Vol:{t['vol_ratio']}×\n\n"
       f"💡 {r}\n\n"
       f"💰 Balance: ${STATE['balance']:.2f}")

    log.info(f"ENTRY {sym} @ ${price} | {asset_type} | score={signal['score']} | sl=${signal['sl']}")

# ═══════════════════════════════════════════════════════
#  NEWS FETCH
# ═══════════════════════════════════════════════════════
gnews_client = GNews(language="en", country="US", max_results=15, period="4h")

def fetch_news_triggered() -> dict:
    triggered = {}
    queries = ["stock market today","federal reserve","crypto bitcoin","tech stocks","earnings"]
    for q in queries:
        try:
            results = gnews_client.get_news(q) or []
            for item in results:
                title = (item.get("title","") or "").lower()
                for kw, syms in NEWS_TRIGGERS.items():
                    if kw in title:
                        for s in syms:
                            triggered.setdefault(s,[]).append(item)
        except: pass
        time.sleep(0.5)
    return triggered

# ═══════════════════════════════════════════════════════
#  MAIN SCAN
# ═══════════════════════════════════════════════════════
def run_scan():
    STATE["scan_count"] += 1
    session = get_market_session()
    now_et  = datetime.now(ET)

    # Reset daily PnL
    today = now_et.strftime("%Y-%m-%d")
    if STATE["daily_reset"] != today:
        STATE["daily_pnl"]  = 0.0
        STATE["daily_reset"] = today

    log.info(f"\n{'='*55}")
    log.info(f"SCAN #{STATE['scan_count']} | {session['label']}")
    log.info(f"Positions: {len(STATE['positions'])} | Balance: ${STATE['balance']:.2f} | P&L: {STATE['total_pnl']:+.2f}")

    # ── Session announcement on change ────────────────
    if STATE["last_session"] != session["session"]:
        STATE["last_session"] = session["session"]
        extras = ""
        if session["session"] == "OPEN":
            extras = f"\n\n🟢 Stocks trading ACTIVE\n📋 Pending signals: {len(STATE.get('pending_signals',{}))}"
        elif session["session"] in ("CLOSED","WEEKEND","HOLIDAY"):
            extras = f"\n\n📅 Next open: {fmt_next_open(session)}\n🔄 Crypto trading continues 24/7"
        elif session["session"] == "PRE":
            extras = "\n\n👁 Building watchlist for open\n⏳ No new stock entries yet"

        tg(f"🕐 <b>SESSION: {session['session']}</b>\n{session['label']}{extras}")

    # ── Always check exits (24/7) ──────────────────────
    check_exits(session)

    # ── Macro (every 4 scans) ──────────────────────────
    if STATE["scan_count"] % 4 == 1:
        macro = fetch_macro()
        log.info(f"📊 Macro: {macro.get('regime')} VIX={macro.get('vix')} DXY={macro.get('dxy')}")
    else:
        macro = STATE.get("macro_context", {"regime":"NEUTRAL","favors":["SPY"],"avoids":[]})

    # ── Fetch news ─────────────────────────────────────
    log.info("📰 Fetching news...")
    triggered = fetch_news_triggered()
    log.info(f"   Triggered: {list(triggered.keys())}")

    # ── PRE-MARKET: build pending signals for open ─────
    if session["session"] == "PRE":
        log.info("⏳ Pre-market: building watchlist for open...")
        for sym in list(STOCK_WATCHLIST.keys()):
            news = triggered.get(sym, [])
            sig  = generate_signal(sym, news, macro, "stock")
            if sig["action"] == "BUY":
                STATE["pending_signals"][sym] = sig
                log.info(f"   📋 Queued {sym}: score={sig['score']}")
        if STATE["pending_signals"]:
            pending_str = "\n".join([f"  {s}: {v['score']:.0f}/100" for s,v in STATE["pending_signals"].items()])
            tg(f"📋 <b>PRE-MARKET WATCHLIST</b>\n\nReady to enter at 9:30 ET:\n{pending_str}\n\nFinal check at open.")

    # ── MARKET OPEN: enter stocks + fire pending ───────
    elif session["session"] == "OPEN":
        # Fire pending signals first
        for sym, sig in list(STATE.get("pending_signals",{}).items()):
            if sym not in STATE["positions"]:
                # Re-validate at open
                tech = get_technicals(sym)
                if tech and sig["score"] >= C["MIN_COMPOSITE_SCORE"]:
                    sig["price"] = tech["price"]  # refresh price
                    enter_position(sym, sig, "stock")
                del STATE["pending_signals"][sym]

        # Live scan during open
        priority = set(triggered.keys()) | set(macro.get("favors",[]))
        for sym in priority:
            if sym in STOCK_WATCHLIST and sym not in STATE["positions"]:
                news = triggered.get(sym,[])
                sig  = generate_signal(sym, news, macro, "stock")
                log.info(f"   {sym}: {sig['action']} score={sig['score']}")
                if sig["action"] == "BUY" and sig["score"] >= C["MIN_COMPOSITE_SCORE"]:
                    enter_position(sym, sig, "stock")
            time.sleep(0.5)

    # ── CRYPTO: always scan (24/7) ─────────────────────
    crypto_triggered = {s:v for s,v in triggered.items() if s in CRYPTO_WATCHLIST}
    crypto_priority  = set(crypto_triggered.keys()) | {s for s in macro.get("favors",[]) if s in CRYPTO_WATCHLIST}

    for sym in crypto_priority:
        if sym in CRYPTO_WATCHLIST and sym not in STATE["positions"]:
            news = crypto_triggered.get(sym,[])
            sig  = generate_signal(sym, news, macro, "crypto")
            log.info(f"   {sym}: {sig['action']} score={sig['score']}")
            if sig["action"] == "BUY" and sig["score"] >= C["MIN_COMPOSITE_SCORE"]:
                enter_position(sym, sig, "crypto")
        time.sleep(0.5)

    # ── Summary every 6 scans ─────────────────────────
    if STATE["scan_count"] % 6 == 0:
        wr = round(STATE["wins"]/(STATE["wins"]+STATE["losses"])*100) if STATE["wins"]+STATE["losses"]>0 else 0
        pos_lines = ""
        for sym,pos in STATE["positions"].items():
            held = round((time.time()-pos["entry_ts"])/3600,1)
            pos_lines += f"\n  {sym}: {pos.get('current_pct',0):+.2f}% | {held}h | SL ${pos['sl']}"
        tg(f"📊 <b>UPDATE</b> — {session['label']}\n\n"
           f"💰 Balance: ${STATE['balance']:.2f}\n"
           f"📈 Total P&L: {'+' if STATE['total_pnl']>=0 else ''}${STATE['total_pnl']:.2f}\n"
           f"📅 Daily P&L: {'+' if STATE['daily_pnl']>=0 else ''}${STATE['daily_pnl']:.2f}\n"
           f"🏆 WR: {wr}% ({STATE['wins']}W/{STATE['losses']}L)\n"
           f"📂 Positions: {len(STATE['positions'])}\n"
           f"📊 {macro.get('regime')} | VIX {macro.get('vix','?')}"
           f"{pos_lines}")

    save_state()
    return session["scan_interval"]

# ═══════════════════════════════════════════════════════
#  BOOT
# ═══════════════════════════════════════════════════════
def main():
    print("""
╔══════════════════════════════════════════════════╗
║   FINCEPT INTELLIGENCE TRADER v2                 ║
║   Swing + Positional | Market-Hours Aware        ║
╚══════════════════════════════════════════════════╝""")

    load_state()

    session = get_market_session()
    print(f"\nCurrent session: {session['session']} — {session['label']}")
    print(f"Stocks:  {'✅ TRADING' if session['can_trade_stocks'] else '⛔ CLOSED'}")
    print(f"Crypto:  {'✅ TRADING (24/7)' if session['can_trade_crypto'] else '⛔'}")
    if session.get("next_open"):
        print(f"Next US market open: {fmt_next_open(session)}")

    tg(f"🚀 <b>FINCEPT TRADER v2 STARTED</b> {'📝' if C['PAPER_MODE'] else '💰'}\n\n"
       f"Session: {session['session']} — {session['label']}\n"
       f"Balance: ${STATE['balance']:.2f}\n\n"
       f"✅ Stocks trade Mon–Fri 9:30–16:00 ET\n"
       f"✅ Crypto trades 24/7\n"
       f"✅ Pre-market builds watchlist\n"
       f"✅ Overnight holds allowed (swing)\n"
       f"✅ SL/TP1/TP2/TP3 ladder active\n\n"
       f"{'Next open: ' + fmt_next_open(session) if session.get('next_open') else 'Market is OPEN now'}")

    while True:
        try:
            interval = run_scan()
        except Exception as e:
            log.error(f"Scan error: {e}")
            interval = 300
        log.info(f"⏱ Sleeping {interval}s ({interval//60}min)...")
        time.sleep(interval)

if __name__ == "__main__":
    main()
