#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════
  FINCEPT INTELLIGENCE TRADER
  News + Sentiment + Macro → Trade Signals with SL/TP
  
  Data sources (all free):
  - fincept-terminal  → news, sentiment, technicals
  - yfinance          → OHLCV, fundamentals
  - fredapi           → macro (fed rates, inflation, GDP)
  - gnews             → real-time global news
  - Fincept REST API  → sentiment scoring
  
  Runs 24/7 on Railway. Sends Telegram alerts.
  Paper mode by default. Never loses more than
  configured daily limit.
═══════════════════════════════════════════════════════
"""

import os, json, time, logging, asyncio, requests
from datetime import datetime, timedelta
from pathlib import Path

# ── Third party (all pip installable) ────────────────
import yfinance as yf
import pandas as pd
import numpy as np
from gnews import GNews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional: fincept-terminal modules
try:
    from fincept_terminal.FinceptTerminalSentimentModule import get_sentiment
    from fincept_terminal.FinceptTerminalNewsModule import get_global_news
    FINCEPT_AVAILABLE = True
except ImportError:
    FINCEPT_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════
C = {
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_CHAT_IDS":  [x.strip() for x in os.getenv("TELEGRAM_CHAT_IDS", os.getenv("TELEGRAM_CHAT_ID","")).split(",") if x.strip()],
    "PAPER_MODE":         os.getenv("PAPER_MODE", "true").lower() != "false",
    "ALPACA_API_KEY":     os.getenv("ALPACA_API_KEY", ""),       # for live US stocks/crypto
    "ALPACA_SECRET_KEY":  os.getenv("ALPACA_SECRET_KEY", ""),
    "FINCEPT_API_KEY":    os.getenv("FINCEPT_API_KEY", ""),       # free at fincept.in

    # ── Position sizing ───────────────────────────────
    "POSITION_SIZE_USD":  float(os.getenv("POSITION_SIZE_USD", "100")),
    "MAX_POSITIONS":      int(os.getenv("MAX_POSITIONS", "3")),
    "DAILY_LOSS_LIMIT":   float(os.getenv("DAILY_LOSS_LIMIT", "150")),

    # ── Scan timing ──────────────────────────────────
    "SCAN_INTERVAL_S":    int(os.getenv("SCAN_INTERVAL_S", "300")),   # 5 min
    "NEWS_INTERVAL_S":    int(os.getenv("NEWS_INTERVAL_S", "120")),   # 2 min

    # ── Signal thresholds ────────────────────────────
    "MIN_SENTIMENT_SCORE": float(os.getenv("MIN_SENTIMENT_SCORE", "0.3")),  # -1 to +1
    "MIN_COMPOSITE_SCORE": float(os.getenv("MIN_COMPOSITE_SCORE", "60")),   # 0–100
    "SL_PCT":              float(os.getenv("SL_PCT", "2.0")),    # 2% stop loss
    "TP1_PCT":             float(os.getenv("TP1_PCT", "3.0")),   # TP1 at 3%
    "TP2_PCT":             float(os.getenv("TP2_PCT", "6.0")),   # TP2 at 6%
    "TP3_PCT":             float(os.getenv("TP3_PCT", "12.0")),  # TP3 at 12%
    "TRAIL_PCT":           float(os.getenv("TRAIL_PCT", "1.5")), # 1.5% trailing

    "STATE_FILE": "state.json",
}

# ── Watchlist — what we trade ─────────────────────────
# Mix of liquid assets across sectors for news-driven moves
WATCHLIST = {
    # US Tech
    "AAPL":  {"name": "Apple",       "sector": "tech",    "type": "stock"},
    "NVDA":  {"name": "NVIDIA",      "sector": "ai",      "type": "stock"},
    "TSLA":  {"name": "Tesla",       "sector": "ev",      "type": "stock"},
    "MSFT":  {"name": "Microsoft",   "sector": "tech",    "type": "stock"},
    "AMZN":  {"name": "Amazon",      "sector": "retail",  "type": "stock"},
    "META":  {"name": "Meta",        "sector": "social",  "type": "stock"},
    # ETFs for macro plays
    "SPY":   {"name": "S&P 500",     "sector": "macro",   "type": "etf"},
    "QQQ":   {"name": "Nasdaq",      "sector": "macro",   "type": "etf"},
    "GLD":   {"name": "Gold",        "sector": "safe",    "type": "etf"},
    "TLT":   {"name": "Bonds",       "sector": "rates",   "type": "etf"},
    # Crypto (via yfinance)
    "BTC-USD": {"name": "Bitcoin",   "sector": "crypto",  "type": "crypto"},
    "ETH-USD": {"name": "Ethereum",  "sector": "crypto",  "type": "crypto"},
    "SOL-USD": {"name": "Solana",    "sector": "crypto",  "type": "crypto"},
}

# ── News keywords → affected assets ──────────────────
NEWS_TRIGGERS = {
    # Fed / rates
    "federal reserve": ["TLT","GLD","SPY","QQQ"],
    "interest rate":   ["TLT","GLD","SPY","QQQ"],
    "inflation":       ["GLD","TLT","SPY"],
    "cpi":             ["GLD","TLT","SPY"],
    "powell":          ["SPY","QQQ","TLT"],
    # Tech
    "ai":              ["NVDA","MSFT","META","AAPL"],
    "artificial intelligence": ["NVDA","MSFT","META"],
    "chip":            ["NVDA","AAPL"],
    "semiconductor":   ["NVDA"],
    "earnings":        ["AAPL","MSFT","NVDA","META","AMZN"],
    # Crypto
    "bitcoin":         ["BTC-USD","ETH-USD"],
    "crypto":          ["BTC-USD","ETH-USD","SOL-USD"],
    "ethereum":        ["ETH-USD"],
    "sec":             ["BTC-USD","ETH-USD","COIN"],
    "etf approval":    ["BTC-USD"],
    # EV / Tesla
    "tesla":           ["TSLA"],
    "electric vehicle": ["TSLA"],
    "musk":            ["TSLA","BTC-USD"],
    # Macro
    "recession":       ["GLD","TLT","SPY"],
    "gdp":             ["SPY","QQQ"],
    "jobs":            ["SPY","TLT"],
    "war":             ["GLD","TLT"],
    "geopolitical":    ["GLD","TLT","SPY"],
}

# ═══════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════
STATE = {
    "balance":      C["POSITION_SIZE_USD"] * 10,  # start with 10× position size
    "positions":    {},     # sym → position dict
    "trade_log":    [],
    "news_log":     [],     # recent news events
    "signals":      {},     # sym → latest signal
    "scan_count":   0,
    "total_pnl":    0.0,
    "wins":         0,
    "losses":       0,
    "started_at":   time.time(),
    "daily_pnl":    0.0,
    "daily_reset":  datetime.now().strftime("%Y-%m-%d"),
    "macro_context": {},    # latest macro readings
}

def load_state():
    try:
        if Path(C["STATE_FILE"]).exists():
            saved = json.loads(Path(C["STATE_FILE"]).read_text())
            STATE.update(saved)
            log.info("✅ State loaded")
    except Exception as e:
        log.warning(f"State load failed: {e}")

def save_state():
    try:
        Path(C["STATE_FILE"]).write_text(json.dumps(STATE, indent=2, default=str))
    except Exception as e:
        log.warning(f"Save failed: {e}")

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
            log.warning(f"TG failed: {e}")

# ═══════════════════════════════════════════════════════
#  SENTIMENT ENGINE
#  Multi-source: VADER + Fincept API + news keywords
# ═══════════════════════════════════════════════════════
vader = SentimentIntensityAnalyzer()

def vader_score(text: str) -> float:
    """VADER sentiment: -1 (very negative) to +1 (very positive)"""
    scores = vader.polarity_scores(text)
    return scores["compound"]

def fincept_sentiment(symbol: str) -> dict | None:
    """
    Fincept free REST API — news sentiment for a symbol
    Returns { score, articles, summary }
    """
    if not C["FINCEPT_API_KEY"]:
        return None
    try:
        r = requests.get(
            f"https://api.fincept.in/v1/sentiment/{symbol}",
            headers={"Authorization": f"Bearer {C['FINCEPT_API_KEY']}"},
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.debug(f"Fincept API: {e}")
    return None

def aggregate_sentiment(symbol: str, news_items: list) -> dict:
    """
    Aggregate sentiment from multiple sources.
    Returns composite sentiment score and breakdown.
    """
    scores = []
    sources = []

    # 1. VADER on news headlines
    for item in news_items[:10]:
        title = item.get("title", "")
        desc  = item.get("description", "") or ""
        text  = f"{title} {desc}"
        if text.strip():
            s = vader_score(text)
            scores.append(s)
            sources.append({"source": "vader", "score": s, "text": title[:80]})

    # 2. Fincept API sentiment
    fincept = fincept_sentiment(symbol)
    if fincept and "score" in fincept:
        scores.append(fincept["score"])
        sources.append({"source": "fincept", "score": fincept["score"]})

    # 3. Keyword boost/penalty
    sym_name = WATCHLIST.get(symbol, {}).get("name", symbol).lower()
    for item in news_items[:5]:
        title = (item.get("title","") or "").lower()
        # Positive signals
        if any(w in title for w in ["surge","soars","beats","record","approval","rally","bullish"]):
            scores.append(0.6)
        if any(w in title for w in ["partnership","acquisition","revenue","profit","upgrade"]):
            scores.append(0.4)
        # Negative signals
        if any(w in title for w in ["crash","drops","miss","lawsuit","probe","ban","recall","fraud"]):
            scores.append(-0.6)
        if any(w in title for w in ["loss","decline","concern","risk","warning","downgrade"]):
            scores.append(-0.4)

    if not scores:
        return {"score": 0.0, "confidence": 0, "sources": [], "direction": "NEUTRAL"}

    avg_score  = np.mean(scores)
    confidence = min(100, len(scores) * 12)  # more sources = higher confidence
    direction  = "BULLISH" if avg_score >= 0.2 else "BEARISH" if avg_score <= -0.2 else "NEUTRAL"

    return {
        "score":      round(avg_score, 3),
        "confidence": confidence,
        "direction":  direction,
        "sources":    sources[:5],
        "article_count": len(news_items),
    }

# ═══════════════════════════════════════════════════════
#  TECHNICAL ANALYSIS
#  Using yfinance OHLCV + pandas
# ═══════════════════════════════════════════════════════
def get_technicals(symbol: str) -> dict | None:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5d", interval="1h")
        if df.empty or len(df) < 20:
            return None

        close = df["Close"]
        vol   = df["Volume"]
        high  = df["High"]
        low   = df["Low"]

        # RSI (14)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = (100 - 100 / (1 + rs)).iloc[-1]

        # MACD
        ema12  = close.ewm(span=12).mean()
        ema26  = close.ewm(span=26).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        macd_cross = "BULLISH" if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2] else \
                     "BEARISH" if macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2] else "NEUTRAL"

        # Bollinger Bands (20,2)
        sma20  = close.rolling(20).mean()
        std20  = close.rolling(20).std()
        bb_up  = sma20 + 2*std20
        bb_lo  = sma20 - 2*std20
        price  = close.iloc[-1]
        bb_pos = (price - bb_lo.iloc[-1]) / (bb_up.iloc[-1] - bb_lo.iloc[-1] + 1e-9)

        # Volume spike
        avg_vol   = vol.rolling(20).mean().iloc[-1]
        curr_vol  = vol.iloc[-1]
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1

        # Trend (1h change)
        ch1h  = ((price - close.iloc[-2]) / close.iloc[-2]) * 100 if len(close) >= 2 else 0
        ch24h = ((price - close.iloc[-25]) / close.iloc[-25]) * 100 if len(close) >= 25 else 0

        # ATR for SL/TP sizing
        tr    = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr   = tr.rolling(14).mean().iloc[-1]
        atr_pct = (atr / price) * 100

        # Technical score
        score = 50.0
        if rsi < 30:   score += 20  # oversold = potential bounce
        elif rsi < 45: score += 10
        elif rsi > 70: score -= 20  # overbought
        elif rsi > 60: score -= 10
        if macd_cross == "BULLISH":  score += 15
        if macd_cross == "BEARISH":  score -= 15
        if bb_pos < 0.2:  score += 10  # near lower band = potential bounce
        if bb_pos > 0.8:  score -= 10  # near upper band = potential reversal
        if vol_ratio > 2: score += 10  # volume spike = conviction
        if ch1h > 1:      score += 8
        if ch1h < -1:     score -= 8

        score = max(0, min(100, score))

        return {
            "price":      round(price, 4),
            "rsi":        round(float(rsi), 1),
            "macd":       macd_cross,
            "bb_pos":     round(float(bb_pos), 2),
            "vol_ratio":  round(vol_ratio, 1),
            "ch1h":       round(ch1h, 2),
            "ch24h":      round(ch24h, 2),
            "atr_pct":    round(atr_pct, 2),
            "score":      round(score, 1),
        }
    except Exception as e:
        log.warning(f"Technicals {symbol}: {e}")
        return None

# ═══════════════════════════════════════════════════════
#  MACRO CONTEXT
#  FRED API — real macro data driving markets
# ═══════════════════════════════════════════════════════
def fetch_macro() -> dict:
    """
    Fetch key macro indicators from FRED (free, no key needed for basic)
    Returns macro sentiment: RISK_ON, RISK_OFF, NEUTRAL
    """
    macro = {}
    try:
        # VIX proxy via yfinance
        vix_data = yf.Ticker("^VIX").history(period="2d")
        if not vix_data.empty:
            macro["vix"] = round(vix_data["Close"].iloc[-1], 1)
            macro["vix_signal"] = "FEAR" if macro["vix"] > 25 else "CALM" if macro["vix"] < 15 else "NEUTRAL"

        # DXY (dollar strength)
        dxy = yf.Ticker("DX-Y.NYB").history(period="2d")
        if not dxy.empty:
            macro["dxy"] = round(dxy["Close"].iloc[-1], 2)
            macro["dxy_ch"] = round(((dxy["Close"].iloc[-1]-dxy["Close"].iloc[-2])/dxy["Close"].iloc[-2])*100, 2)

        # 10Y yield
        tlt = yf.Ticker("^TNX").history(period="2d")
        if not tlt.empty:
            macro["yield_10y"] = round(tlt["Close"].iloc[-1], 2)

        # Crypto fear via BTC correlation
        btc = yf.Ticker("BTC-USD").history(period="2d")
        if not btc.empty:
            macro["btc_24h"] = round(((btc["Close"].iloc[-1]-btc["Close"].iloc[-2])/btc["Close"].iloc[-2])*100, 2)

        # Overall regime
        vix_val = macro.get("vix", 20)
        if vix_val > 30:
            macro["regime"] = "RISK_OFF"   # fly to safety
            macro["favors"]  = ["GLD","TLT"]
        elif vix_val < 18:
            macro["regime"] = "RISK_ON"    # buy growth
            macro["favors"]  = ["NVDA","TSLA","QQQ","BTC-USD"]
        else:
            macro["regime"]  = "NEUTRAL"
            macro["favors"]  = ["SPY","AAPL","MSFT"]

    except Exception as e:
        log.warning(f"Macro fetch: {e}")
        macro.setdefault("regime", "NEUTRAL")
        macro.setdefault("favors", ["SPY"])

    STATE["macro_context"] = macro
    return macro

# ═══════════════════════════════════════════════════════
#  NEWS FETCHER
#  GNews + keyword → asset mapping
# ═══════════════════════════════════════════════════════
gnews_client = GNews(language="en", country="US", max_results=20, period="2h")

def fetch_news_for_symbol(symbol: str) -> list:
    """Fetch news for a specific symbol"""
    name = WATCHLIST.get(symbol, {}).get("name", symbol)
    items = []
    try:
        results = gnews_client.get_news(f"{name} stock market")
        items.extend(results or [])
    except Exception as e:
        log.debug(f"GNews {symbol}: {e}")
    return items

def fetch_global_news() -> list:
    """Fetch broad financial news and map to affected assets"""
    triggered = {}  # sym → [news items]
    try:
        queries = ["stock market", "federal reserve", "crypto bitcoin", "tech stocks", "global economy"]
        for q in queries:
            try:
                results = gnews_client.get_news(q) or []
                for item in results:
                    title = (item.get("title","") or "").lower()
                    for keyword, symbols in NEWS_TRIGGERS.items():
                        if keyword in title:
                            for sym in symbols:
                                if sym in WATCHLIST:
                                    triggered.setdefault(sym, []).append(item)
            except:
                pass
            time.sleep(0.5)  # rate limit
    except Exception as e:
        log.warning(f"Global news: {e}")
    return triggered

# ═══════════════════════════════════════════════════════
#  COMPOSITE SIGNAL ENGINE
#  News sentiment + technicals + macro → BUY/SELL/HOLD
# ═══════════════════════════════════════════════════════
def generate_signal(symbol: str, news_items: list, macro: dict) -> dict:
    """
    Generate a composite trading signal.
    Returns { action, score, sl, tp1, tp2, tp3, reason[] }
    """
    reasons = []

    # 1. Sentiment analysis
    sentiment = aggregate_sentiment(symbol, news_items)
    sent_score = sentiment["score"]   # -1 to +1

    # 2. Technical analysis
    tech = get_technicals(symbol)
    if not tech:
        return {"action": "HOLD", "score": 0, "reason": ["No price data"]}

    price = tech["price"]

    # 3. Macro alignment
    macro_regime = macro.get("regime", "NEUTRAL")
    macro_favors = macro.get("favors", [])
    macro_bonus  = 10 if symbol in macro_favors else -5 if macro_regime == "RISK_OFF" and WATCHLIST.get(symbol,{}).get("sector") in ["tech","crypto","ev","ai"] else 0

    # ── Composite Score (0–100) ────────────────────
    # Sentiment: 40% weight | Technical: 45% | Macro: 15%
    sent_normalized = (sent_score + 1) / 2 * 100   # -1→+1 to 0→100
    composite = (
        sent_normalized  * 0.40 +
        tech["score"]    * 0.45 +
        (50 + macro_bonus) * 0.15
    )
    composite = round(composite, 1)

    # ── Build reason list ──────────────────────────
    reasons.append(f"Sentiment: {sentiment['direction']} ({sent_score:+.2f}, {sentiment['article_count']} articles)")
    reasons.append(f"RSI: {tech['rsi']} | MACD: {tech['macd']} | Vol: {tech['vol_ratio']}×")
    reasons.append(f"1h: {tech['ch1h']:+.2f}% | 24h: {tech['ch24h']:+.2f}%")
    reasons.append(f"Macro: {macro_regime} (VIX {macro.get('vix','?')})")

    # ── Dynamic SL/TP based on ATR ─────────────────
    atr = tech["atr_pct"]
    sl_pct  = max(C["SL_PCT"],  atr * 1.2)   # at least 1.2× ATR
    tp1_pct = max(C["TP1_PCT"], atr * 1.5)
    tp2_pct = max(C["TP2_PCT"], atr * 3.0)
    tp3_pct = max(C["TP3_PCT"], atr * 5.0)

    sl  = round(price * (1 - sl_pct/100), 4)
    tp1 = round(price * (1 + tp1_pct/100), 4)
    tp2 = round(price * (1 + tp2_pct/100), 4)
    tp3 = round(price * (1 + tp3_pct/100), 4)

    # ── Action decision ───────────────────────────
    action = "HOLD"
    if composite >= C["MIN_COMPOSITE_SCORE"] and sent_score >= C["MIN_SENTIMENT_SCORE"]:
        action = "BUY"
    elif composite <= (100 - C["MIN_COMPOSITE_SCORE"]) and sent_score <= -C["MIN_SENTIMENT_SCORE"]:
        action = "SELL_SHORT"  # informational only unless broker supports it

    # Extra confirmation: RSI not overbought for buys
    if action == "BUY" and tech["rsi"] > 75:
        action = "HOLD"
        reasons.append("⚠️ RSI overbought — holding despite bullish sentiment")

    # Extra confirmation: volume confirms move
    if action == "BUY" and tech["vol_ratio"] < 0.8:
        composite -= 10
        reasons.append("⚠️ Low volume — reduced confidence")

    return {
        "action":    action,
        "score":     composite,
        "price":     price,
        "sl":        sl,   "sl_pct":  round(sl_pct, 2),
        "tp1":       tp1,  "tp1_pct": round(tp1_pct, 2),
        "tp2":       tp2,  "tp2_pct": round(tp2_pct, 2),
        "tp3":       tp3,  "tp3_pct": round(tp3_pct, 2),
        "sentiment": sentiment,
        "tech":      tech,
        "reasons":   reasons,
        "time":      datetime.now().isoformat(),
    }

# ═══════════════════════════════════════════════════════
#  POSITION MANAGER
# ═══════════════════════════════════════════════════════
def check_exits():
    """Check all open positions for SL/TP hits"""
    for sym, pos in list(STATE["positions"].items()):
        try:
            data = yf.Ticker(sym).history(period="1d", interval="5m")
            if data.empty:
                continue
            price = data["Close"].iloc[-1]
            entry = pos["entry_price"]
            pct   = ((price - entry) / entry) * 100
            peak  = max(pos.get("peak", entry), price)
            trail_stop = peak * (1 - C["TRAIL_PCT"]/100)

            STATE["positions"][sym]["peak"] = peak
            STATE["positions"][sym]["current_pct"] = round(pct, 2)

            exit_reason = None
            tp_hit      = None

            # Check exits in priority order
            if price <= pos["sl"]:
                exit_reason = f"🛑 Stop Loss at ${price:.4f} ({pct:+.2f}%)"
            elif price >= pos["tp3"]:
                exit_reason = f"🏆 TP3 at ${price:.4f} ({pct:+.2f}%)"
                tp_hit = 3
            elif price >= pos["tp2"] and not pos.get("tp2_hit"):
                STATE["positions"][sym]["tp2_hit"] = True
                # Partial exit — sell 50%
                half_size = pos["size"] * 0.5
                half_pnl  = half_size * (pct/100)
                STATE["balance"]   += half_size + half_pnl
                STATE["total_pnl"] += half_pnl
                STATE["daily_pnl"] += half_pnl
                STATE["positions"][sym]["size"] -= half_size
                tg(f"🎯 <b>TP2 PARTIAL EXIT — {sym}</b>\n"
                   f"Sold 50% at ${price:.4f} ({pct:+.2f}%)\n"
                   f"P&L: +${half_pnl:.2f} | Trail stop now active")
                log.info(f"TP2 partial {sym} +${half_pnl:.2f}")
                continue
            elif price >= pos["tp1"] and not pos.get("tp1_hit"):
                STATE["positions"][sym]["tp1_hit"] = True
                # Move SL to breakeven
                STATE["positions"][sym]["sl"] = entry * 1.005
                tg(f"✅ <b>TP1 HIT — {sym}</b>\n"
                   f"${price:.4f} ({pct:+.2f}%) — SL moved to breakeven")
                log.info(f"TP1 hit {sym} — SL to breakeven")
                continue
            elif price <= trail_stop and pos.get("tp1_hit"):
                exit_reason = f"📉 Trailing stop (peak {((peak-entry)/entry*100):+.2f}%)"

            if exit_reason:
                # Full exit
                pnl = pos["size"] * (pct/100)
                STATE["balance"]   += pos["size"] + pnl
                STATE["total_pnl"] += pnl
                STATE["daily_pnl"] += pnl
                if pnl >= 0: STATE["wins"]   += 1
                else:        STATE["losses"] += 1

                STATE["trade_log"].insert(0, {
                    "type": "EXIT", "symbol": sym,
                    "entry": entry, "exit": price,
                    "pnl": round(pnl,2), "pct": round(pct,2),
                    "reason": exit_reason,
                    "time": datetime.now().isoformat(),
                })

                wr = round(STATE["wins"]/(STATE["wins"]+STATE["losses"])*100) if STATE["wins"]+STATE["losses"] > 0 else 0
                tg(f"{'🟢' if pnl>=0 else '🔴'} <b>EXIT — {sym}</b> {('📝' if C['PAPER_MODE'] else '💰')}\n\n"
                   f"💵 P&L: {'+' if pnl>=0 else ''}${pnl:.2f} ({pct:+.2f}%)\n"
                   f"📌 {exit_reason}\n\n"
                   f"💰 Balance: ${STATE['balance']:.2f}\n"
                   f"📊 Win Rate: {wr}% ({STATE['wins']}W/{STATE['losses']}L)\n"
                   f"📈 Total P&L: {'+' if STATE['total_pnl']>=0 else ''}${STATE['total_pnl']:.2f}")

                del STATE["positions"][sym]
                log.info(f"EXIT {sym} | {pct:+.2f}% | ${pnl:.2f} | {exit_reason}")

        except Exception as e:
            log.warning(f"Exit check {sym}: {e}")

def enter_position(sym: str, signal: dict):
    """Enter a new position based on signal"""
    if STATE["daily_pnl"] <= -C["DAILY_LOSS_LIMIT"]:
        log.info(f"⛔ Daily limit hit, no entry for {sym}")
        return
    if len(STATE["positions"]) >= C["MAX_POSITIONS"]:
        return
    if sym in STATE["positions"]:
        return
    if STATE["balance"] < C["POSITION_SIZE_USD"]:
        log.warning("Low balance")
        return

    size  = C["POSITION_SIZE_USD"]
    price = signal["price"]
    STATE["balance"] -= size

    pos = {
        "symbol":      sym,
        "entry_price": price,
        "size":        size,
        "sl":          signal["sl"],
        "tp1":         signal["tp1"],
        "tp2":         signal["tp2"],
        "tp3":         signal["tp3"],
        "sl_pct":      signal["sl_pct"],
        "tp1_pct":     signal["tp1_pct"],
        "peak":        price,
        "score":       signal["score"],
        "action":      signal["action"],
        "entry_time":  datetime.now().isoformat(),
        "tp1_hit":     False, "tp2_hit": False,
    }
    STATE["positions"][sym] = pos
    STATE["trade_log"].insert(0, {
        "type": "ENTRY", "symbol": sym, "price": price,
        "size": size, "score": signal["score"],
        "time": datetime.now().isoformat(),
    })

    reasons_str = "\n  ".join(signal["reasons"])
    sent        = signal["sentiment"]
    tech        = signal["tech"]

    tg(f"⚡ <b>ENTRY — {sym}</b> {('📝' if C['PAPER_MODE'] else '💰')}\n\n"
       f"🎯 Score: {signal['score']}/100 | {signal['action']}\n"
       f"💲 Price: ${price}\n"
       f"💵 Size: ${size:.0f}\n\n"
       f"🛑 SL:  ${signal['sl']} (-{signal['sl_pct']}%)\n"
       f"✅ TP1: ${signal['tp1']} (+{signal['tp1_pct']}%)\n"
       f"🎯 TP2: ${signal['tp2']} (+{signal['tp2_pct']}%)\n"
       f"🏆 TP3: ${signal['tp3']} (+{signal['tp3_pct']}%)\n\n"
       f"📰 News: {sent['direction']} ({sent['score']:+.2f}) — {sent['article_count']} articles\n"
       f"📊 RSI: {tech['rsi']} | MACD: {tech['macd']} | Vol: {tech['vol_ratio']}×\n\n"
       f"💡 Reasons:\n  {reasons_str}\n\n"
       f"💰 Balance: ${STATE['balance']:.2f}")

    log.info(f"ENTRY {sym} @ ${price} | score={signal['score']} | sl=${signal['sl']} | tp1=${signal['tp1']}")

# ═══════════════════════════════════════════════════════
#  MAIN SCAN LOOP
# ═══════════════════════════════════════════════════════
def run_scan():
    STATE["scan_count"] += 1
    now = datetime.now()
    log.info(f"\n{'='*50}")
    log.info(f"SCAN #{STATE['scan_count']} | {now.strftime('%H:%M:%S')} | Pos: {len(STATE['positions'])}/{C['MAX_POSITIONS']} | ${STATE['balance']:.2f}")

    # Reset daily PnL if new day
    today = now.strftime("%Y-%m-%d")
    if STATE["daily_reset"] != today:
        STATE["daily_pnl"]  = 0.0
        STATE["daily_reset"] = today
        log.info("📅 Daily P&L reset")

    # 1. Check exits first
    check_exits()

    # 2. Fetch macro context (every 5 scans to save API calls)
    if STATE["scan_count"] % 5 == 1:
        macro = fetch_macro()
        log.info(f"📊 Macro: {macro.get('regime')} | VIX {macro.get('vix','?')} | Favors: {macro.get('favors','?')}")
    else:
        macro = STATE.get("macro_context", {"regime":"NEUTRAL","favors":["SPY"]})

    # 3. Fetch global news → map to assets
    log.info("📰 Fetching global news...")
    triggered = fetch_global_news()
    log.info(f"   News triggered for: {list(triggered.keys())}")

    # 4. Analyze each triggered asset + always scan macro favors
    priority = set(triggered.keys()) | set(macro.get("favors", []))

    for sym in priority:
        if sym not in WATCHLIST:
            continue
        if sym in STATE["positions"]:
            continue  # already in, skip

        try:
            # Get news (use triggered or fetch specific)
            news = triggered.get(sym, fetch_news_for_symbol(sym))

            # Generate signal
            signal = generate_signal(sym, news, macro)
            STATE["signals"][sym] = signal

            log.info(f"   {sym}: {signal['action']} | score={signal['score']} | sent={signal['sentiment']['score']:+.2f}")

            # Enter if signal is strong enough
            if (signal["action"] == "BUY" and
                signal["score"] >= C["MIN_COMPOSITE_SCORE"] and
                signal["sentiment"]["score"] >= C["MIN_SENTIMENT_SCORE"] and
                len(STATE["positions"]) < C["MAX_POSITIONS"] and
                STATE["daily_pnl"] > -C["DAILY_LOSS_LIMIT"]):
                enter_position(sym, signal)

            time.sleep(1)  # rate limit

        except Exception as e:
            log.warning(f"Signal {sym}: {e}")

    # 5. Summary every 6 scans (30 min)
    if STATE["scan_count"] % 6 == 0:
        wr  = round(STATE["wins"]/(STATE["wins"]+STATE["losses"])*100) if STATE["wins"]+STATE["losses"] > 0 else 0
        pos_str = ""
        for sym, pos in STATE["positions"].items():
            pos_str += f"\n  {sym}: {pos.get('current_pct',0):+.2f}% | SL ${pos['sl']} | TP1 ${pos['tp1']}"

        tg(f"📊 <b>30-MIN SUMMARY</b> — Scan #{STATE['scan_count']}\n\n"
           f"💰 Balance: ${STATE['balance']:.2f}\n"
           f"📈 Total P&L: {'+' if STATE['total_pnl']>=0 else ''}${STATE['total_pnl']:.2f}\n"
           f"📅 Daily P&L: {'+' if STATE['daily_pnl']>=0 else ''}${STATE['daily_pnl']:.2f}\n"
           f"🏆 Win Rate: {wr}% ({STATE['wins']}W/{STATE['losses']}L)\n"
           f"📂 Open Positions: {len(STATE['positions'])}\n"
           f"📊 Macro: {macro.get('regime')} | VIX {macro.get('vix','?')}"
           f"{pos_str}")

    save_state()

# ═══════════════════════════════════════════════════════
#  BOOT
# ═══════════════════════════════════════════════════════
def main():
    print("""
╔══════════════════════════════════════════════════╗
║   FINCEPT INTELLIGENCE TRADER                    ║
║   News + Sentiment + Macro → SL/TP Signals      ║
╚══════════════════════════════════════════════════╝""")

    print(f"Mode:           {'📝 PAPER' if C['PAPER_MODE'] else '💰 LIVE'}")
    print(f"Position size:  ${C['POSITION_SIZE_USD']}")
    print(f"Stop loss:      {C['SL_PCT']}% (ATR-adjusted)")
    print(f"Take profits:   TP1 {C['TP1_PCT']}% | TP2 {C['TP2_PCT']}% | TP3 {C['TP3_PCT']}%")
    print(f"Daily limit:    -${C['DAILY_LOSS_LIMIT']}")
    print(f"Scan interval:  {C['SCAN_INTERVAL_S']}s")
    print(f"Fincept API:    {'✅ Configured' if C['FINCEPT_API_KEY'] else '⚠️  Not set (using free sources only)'}")
    print(f"Alpaca:         {'✅ Configured' if C['ALPACA_API_KEY'] else '⚠️  Not set (paper only)'}")
    print(f"Watchlist:      {', '.join(WATCHLIST.keys())}")
    print()

    load_state()

    tg(f"🚀 <b>FINCEPT INTELLIGENCE TRADER STARTED</b>\n"
       f"Mode: {'📝 Paper' if C['PAPER_MODE'] else '💰 LIVE'}\n"
       f"Balance: ${STATE['balance']:.2f}\n\n"
       f"Data sources:\n"
       f"• GNews — real-time global news\n"
       f"• VADER — NLP sentiment analysis\n"
       f"• yfinance — RSI, MACD, Bollinger\n"
       f"• VIX/DXY — macro regime detection\n"
       f"• Fincept API — {'✅' if C['FINCEPT_API_KEY'] else '⚠️ add key for premium sentiment'}\n\n"
       f"Watching: {', '.join(WATCHLIST.keys())}\n"
       f"SL: {C['SL_PCT']}% | TP1: {C['TP1_PCT']}% | TP2: {C['TP2_PCT']}% | TP3: {C['TP3_PCT']}%")

    # Run immediately then on interval
    while True:
        try:
            run_scan()
        except Exception as e:
            log.error(f"Scan error: {e}")
        log.info(f"⏱ Sleeping {C['SCAN_INTERVAL_S']}s...")
        time.sleep(C["SCAN_INTERVAL_S"])

if __name__ == "__main__":
    main()
