from crewai.tools import tool
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import re
from datetime import datetime, timedelta, timezone
from ddgs import DDGS

_NEWS_CUTOFF = datetime.now(timezone.utc) - timedelta(days=30)
_STOCKS_FILE = os.path.join(os.path.dirname(__file__), "stocks.md")
_MEMORY_DIR = os.path.join(os.path.dirname(__file__), "notebook_memory")
_SEC_HEADERS = {"User-Agent": "TheCouncil/1.0 (contact@example.com)"}


# ─────────────────────────────────────────────
# Helper: get CIK from ticker for SEC EDGAR
# ─────────────────────────────────────────────

def _ticker_to_cik(ticker: str) -> int:
    """Convert a ticker symbol to SEC CIK number."""
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=_SEC_HEADERS,
        timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker.upper():
            return int(entry["cik_str"])
    raise ValueError(f"Ticker {ticker} not found in SEC database")


# ─────────────────────────────────────────────
# Tool 1: News Search
# ─────────────────────────────────────────────

@tool("Search Internet for News")
def search_news(query: str) -> str:
    """Use this tool to find the latest news headlines, sources, publication dates, and summaries for any stock ticker or company name. Returns up to 5 recent news items from Yahoo Finance. Each item includes the exact headline, publisher name, publication date, a summary, and a link to the full article. Use this tool FIRST before writing any news section."""
    news_parts = []

    # Yahoo Finance (no API key needed, most reliable)
    try:
        stock = yf.Ticker(query)
        items = stock.news or []
        items.sort(key=lambda x: x.get("content", {}).get("pubDate", ""), reverse=True)
        seen = 0
        for item in items:
            if seen >= 5:
                break
            c = item.get("content", {})
            pub_str = c.get("pubDate", "")
            try:
                pub_date = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                if pub_date < _NEWS_CUTOFF:
                    continue
            except Exception:
                pass
            link = (
                c.get("canonicalUrl", {}).get("url")
                or c.get("clickThroughUrl", {}).get("url")
                or "No URL"
            )
            provider = item.get("provider", {}).get("displayName", "Yahoo Finance")
            news_parts.append(
                f"[Yahoo Finance]\n"
                f"Headline: {c.get('title')}\n"
                f"Source: {provider}\n"
                f"Published: {c.get('pubDate', 'unknown')}\n"
                f"Summary: {c.get('summary', '')}\n"
                f"Link: {link}"
            )
            seen += 1
    except Exception:
        pass

    if not news_parts:
        pass
    else:
        return "\n\n".join(news_parts)

    # Serper API (requires SERPER_API_KEY, 2500 free queries/month)
    serper_key = os.getenv("SERPER_API_KEY")
    if serper_key:
        try:
            resp = requests.post(
                "https://google.serper.dev/news",
                json={"q": query, "num": 5},
                headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            for article in data.get("news", []):
                news_parts.append(
                    f"[Serper News]\n"
                    f"Headline: {article.get('title', 'N/A')}\n"
                    f"Source: {article.get('source', 'Serper')}\n"
                    f"Published: {article.get('date', 'unknown')}\n"
                    f"Summary: {article.get('snippet', '')}\n"
                    f"Link: {article.get('link', 'No URL')}"
                )
        except Exception:
            pass

    if not news_parts:
        pass
    else:
        return "\n\n".join(news_parts)

    # Fallback: DuckDuckGo (free, no API key needed)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=5, timelimit="m"))
        for item in results:
            news_parts.append(
                f"[DuckDuckGo News]\n"
                f"Headline: {item.get('title', 'N/A')}\n"
                f"Source: {item.get('source', 'DuckDuckGo')}\n"
                f"Published: {item.get('date', 'unknown')}\n"
                f"Summary: {item.get('body', '')}\n"
                f"Link: {item.get('url', 'No URL')}"
            )
    except Exception:
        pass

    if not news_parts:
        return "No news found."
    return "\n\n".join(news_parts)


# ─────────────────────────────────────────────
# Tool 2: Technical Analysis
# ─────────────────────────────────────────────

@tool("Fetch Technical Analysis Data")
def fetch_stock_analysis(ticker: str) -> str:
    """Fetches 1 year of price data and calculates all technical indicators."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        if hist.empty:
            return f"Could not fetch data for {ticker}. Check if the ticker symbol is correct."

        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']

        current_price = close.iloc[-1]
        prev_close = close.iloc[-2] if len(close) > 1 else current_price

        # ── Trend ──
        ema_20 = close.ewm(span=20).mean().iloc[-1]
        ema_50 = close.ewm(span=50).mean().iloc[-1]
        ema_200 = close.ewm(span=200).mean().iloc[-1] if len(close) >= 200 else None

        price_vs_ema20 = "above" if current_price > ema_20 else "below"
        price_vs_ema50 = "above" if current_price > ema_50 else "below"
        trend = "Uptrend" if current_price > ema_50 else "Downtrend"

        # ── Support & Resistance (20-day rolling) ──
        support = low.rolling(20).min().iloc[-1]
        resistance = high.rolling(20).max().iloc[-1]
        s2 = low.rolling(50).min().iloc[-1]
        r2 = high.rolling(50).max().iloc[-1]

        # ── RSI 14 ──
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.rolling(14).mean().iloc[-1]
        avg_loss = loss.rolling(14).mean().iloc[-1]
        rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss)) if avg_loss != 0 and not pd.isna(avg_loss) else 50.0

        # ── ATR 14 ──
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        if pd.isna(atr):
            atr = 0.0

        # ── Bollinger Bands (20,2) ──
        bb_sma = close.rolling(20).mean().iloc[-1]
        bb_std = close.rolling(20).std().iloc[-1]
        bb_upper = bb_sma + (bb_std * 2)
        bb_lower = bb_sma - (bb_std * 2)
        bb_width = ((bb_upper - bb_lower) / bb_sma) * 100

        # ── Volatility & Return ──
        daily_returns = close.pct_change()
        daily_mean = daily_returns.mean() * 100
        std_dev = daily_returns.std() * 100
        roi_60d = ((current_price - close.iloc[-60]) / close.iloc[-60]) * 100 if len(close) >= 60 else None
        roi_1y = ((current_price - close.iloc[0]) / close.iloc[0]) * 100

        # ── Sharpe Ratio (1y, using daily returns, risk-free ≈ 0) ──
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0.0

        # ── Fibonacci Retracement (90-day swing) ──
        lookback = min(90, len(close))
        swing_high = close.iloc[-lookback:].max()
        swing_low = close.iloc[-lookback:].min()
        fib_range = swing_high - swing_low
        fib_levels_down = {
            "0.236": swing_high - fib_range * 0.236,
            "0.382": swing_high - fib_range * 0.382,
            "0.500": swing_high - fib_range * 0.500,
            "0.618": swing_high - fib_range * 0.618,
            "0.786": swing_high - fib_range * 0.786,
        }
        fib_levels_up = {
            "0.236": swing_low + fib_range * 0.236,
            "0.382": swing_low + fib_range * 0.382,
            "0.500": swing_low + fib_range * 0.500,
            "0.618": swing_low + fib_range * 0.618,
            "0.786": swing_low + fib_range * 0.786,
        }
        fib_note = "Price is above the 90-day swing midpoint" if current_price > fib_levels_up["0.500"] else "Price is below the 90-day swing midpoint"

        # ── Classic Pivot Points (previous month) ──
        pm = min(21, len(close))
        pp_high = high.iloc[-pm:].max()
        pp_low = low.iloc[-pm:].min()
        pp_close = close.iloc[-1]
        pp = (pp_high + pp_low + pp_close) / 3
        r1 = 2 * pp - pp_low
        r2 = pp + (pp_high - pp_low)
        r3 = pp_high + 2 * (pp - pp_low)
        s1 = 2 * pp - pp_high
        s2 = pp - (pp_high - pp_low)
        s3 = pp_low - 2 * (pp_high - pp_low)

        # ── Volume ──
        avg_vol_20 = volume.rolling(20).mean().iloc[-1]
        latest_vol = volume.iloc[-1]
        vol_ratio = latest_vol / avg_vol_20 if avg_vol_20 != 0 else 1

        # ── Build output ──
        lines = [
            f"--- TECHNICAL DATA FOR {ticker.upper()} ---",
            f"Current Price: ${current_price:.2f}",
            f"Previous Close: ${prev_close:.2f}",
            f"Current Trend: {trend}",
            "",
            "--- MOVING AVERAGES ---",
            f"EMA 20: ${ema_20:.2f} (Price {price_vs_ema20})",
            f"EMA 50: ${ema_50:.2f} (Price {price_vs_ema50})",
        ]
        if ema_200 is not None:
            price_vs_ema200 = "above" if current_price > ema_200 else "below"
            lines.append(f"EMA 200: ${ema_200:.2f} (Price {price_vs_ema200})")
        else:
            lines.append("EMA 200: Insufficient data (< 200 trading days)")

        lines += [
            "",
            "--- SUPPORT & RESISTANCE ---",
            f"Support (20d): ${support:.2f}",
            f"Support (50d): ${s2:.2f}",
            f"Resistance (20d): ${resistance:.2f}",
            f"Resistance (50d): ${r2:.2f}",
            "",
            "--- MOMENTUM ---",
            f"RSI 14: {rsi:.1f}",
            f"ATR 14: ${atr:.2f}",
            "",
            "--- BOLLINGER BANDS (20,2) ---",
            f"Upper Band: ${bb_upper:.2f}",
            f"Middle (SMA): ${bb_sma:.2f}",
            f"Lower Band: ${bb_lower:.2f}",
            f"Band Width: {bb_width:.1f}%",
            "",
            "--- VOLATILITY & RETURNS ---",
            f"Daily Return Mean: {daily_mean:.3f}%",
            f"Std Dev (Volatility): {std_dev:.2f}%",
            f"Sharpe Ratio (1y): {sharpe_ratio:.2f}",
        ]
        if roi_60d is not None:
            lines.append(f"60-Day ROI: {roi_60d:.2f}%")
        lines.append(f"1-Year ROI: {roi_1y:.2f}%")
        lines += [
            "",
            "--- FIBONACCI RETRACEMENT (90-day swing) ---",
            f"Swing High (90d): ${swing_high:.2f}",
            f"Swing Low (90d): ${swing_low:.2f}",
            "Upside Retracement Levels (from low):",
            f"  23.6%: ${fib_levels_up['0.236']:.2f} | 38.2%: ${fib_levels_up['0.382']:.2f} | 50%: ${fib_levels_up['0.500']:.2f} | 61.8%: ${fib_levels_up['0.618']:.2f} | 78.6%: ${fib_levels_up['0.786']:.2f}",
            "Downside Retracement Levels (from high):",
            f"  23.6%: ${fib_levels_down['0.236']:.2f} | 38.2%: ${fib_levels_down['0.382']:.2f} | 50%: ${fib_levels_down['0.500']:.2f} | 61.8%: ${fib_levels_down['0.618']:.2f} | 78.6%: ${fib_levels_down['0.786']:.2f}",
            f"Fibonacci Note: {fib_note}",
            "",
            "--- PIVOT POINTS (classic, monthly) ---",
            f"Pivot Point (PP): ${pp:.2f}",
            f"Resistance: R1=${r1:.2f} | R2=${r2:.2f} | R3=${r3:.2f}",
            f"Support: S1=${s1:.2f} | S2=${s2:.2f} | S3=${s3:.2f}",
            "",
            "--- VOLUME ---",
            f"Latest Volume: {latest_vol:,.0f}",
            f"Avg Volume (20d): {avg_vol_20:,.0f}",
            f"Volume Ratio: {vol_ratio:.2f}x",
        ]

        return "\n".join(lines)

    except Exception as e:
        return f"Error calculating data for {ticker}: {e}"


# ─────────────────────────────────────────────
# Tool 3: Fundamental Data
# ─────────────────────────────────────────────

@tool("Fetch Fundamental Data")
def fetch_fundamentals(ticker: str) -> str:
    """Fetches fundamental data: market cap, earnings date, analyst consensus, financials."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        current_price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose", "N/A")
        market_cap = info.get("marketCap", "N/A")
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")

        # Format market cap
        if market_cap != "N/A":
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_str = f"${market_cap:,.0f}"
        else:
            market_cap_str = "N/A"

        # Earnings date
        earnings_ts = info.get("earningsTimestamp")
        if earnings_ts:
            next_earnings = datetime.fromtimestamp(earnings_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            earnings_calendar = stock.calendar
            if earnings_calendar is not None and not earnings_calendar.empty:
                next_earnings = str(earnings_calendar.index[0].strftime("%Y-%m-%d"))
            else:
                next_earnings = "Data unavailable"

        # Analyst consensus
        rec_key = info.get("recommendationKey", "N/A")
        rec_mean = info.get("recommendationMean", "N/A")
        target_mean = info.get("targetMeanPrice", "N/A")
        target_high = info.get("targetHighPrice", "N/A")
        target_low = info.get("targetLowPrice", "N/A")

        # Financials
        revenue = info.get("totalRevenue", "N/A")
        if revenue != "N/A":
            revenue_str = f"${revenue/1e9:.2f}B" if revenue >= 1e9 else f"${revenue/1e6:.2f}M"
        else:
            revenue_str = "N/A"

        revenue_growth = info.get("revenueGrowth", "N/A")
        if revenue_growth != "N/A":
            revenue_growth = f"{revenue_growth*100:.1f}%"

        eps_ttm = info.get("trailingEps", "N/A")
        eps_fwd = info.get("forwardEps", "N/A")
        fcf = info.get("freeCashflow", "N/A")
        if fcf != "N/A":
            fcf_str = f"${fcf/1e9:.2f}B" if abs(fcf) >= 1e9 else f"${fcf/1e6:.2f}M"
        else:
            fcf_str = "N/A"

        pe_ttm = info.get("trailingPE", "N/A")
        pe_fwd = info.get("forwardPE", "N/A")
        ps = info.get("priceToSalesTrailing12Months", "N/A")
        peg = info.get("pegRatio", "N/A")
        beta = info.get("beta", "N/A")

        # Profitability
        profit_margin = info.get("profitMargins", "N/A")
        if profit_margin != "N/A":
            profit_margin = f"{profit_margin*100:.1f}%"
        op_margin = info.get("operatingMargins", "N/A")
        if op_margin != "N/A":
            op_margin = f"{op_margin*100:.1f}%"
        roe = info.get("returnOnEquity", "N/A")
        if roe != "N/A":
            roe = f"{roe*100:.1f}%"

        # Debt
        de = info.get("debtToEquity", "N/A")
        if de != "N/A":
            de = f"{de:.1f}%"
        current_r = info.get("currentRatio", "N/A")

        lines = [
            f"--- FUNDAMENTAL DATA FOR {ticker.upper()} ---",
            f"Company: {info.get('longName', info.get('shortName', ticker))}",
            f"Sector: {sector}",
            f"Industry: {industry}",
            f"Current Price: ${current_price}" if isinstance(current_price, (int, float)) else f"Current Price: {current_price}",
            f"Market Cap: {market_cap_str}",
            "",
            "--- EARNINGS & CONSENSUS ---",
            f"Next Earnings Date: {next_earnings}",
            f"Analyst Consensus: {rec_key.upper() if rec_key != 'N/A' else 'N/A'}",
            f"Mean Target: ${target_mean}" if target_mean != "N/A" else "Mean Target: N/A",
            f"Target Range: ${target_low} — ${target_high}" if all(v != "N/A" for v in [target_low, target_high]) else "Target Range: N/A",
            "",
            "--- FINANCIALS ---",
            f"Total Revenue (TTM): {revenue_str}",
            f"Revenue Growth (YoY): {revenue_growth}",
            f"EPS (TTM): ${eps_ttm}" if eps_ttm != "N/A" else "EPS (TTM): N/A",
            f"EPS (Forward): ${eps_fwd}" if eps_fwd != "N/A" else "EPS (Forward): N/A",
            f"Free Cash Flow: {fcf_str}",
            "",
            "--- VALUATION ---",
            f"P/E (TTM): {pe_ttm}" if pe_ttm != "N/A" else "P/E (TTM): N/A",
            f"P/E (Forward): {pe_fwd}" if pe_fwd != "N/A" else "P/E (Forward): N/A",
            f"P/S (TTM): {ps}" if ps != "N/A" else "P/S (TTM): N/A",
            f"PEG Ratio: {peg}" if peg != "N/A" else "PEG Ratio: N/A",
            f"Beta: {beta}" if beta != "N/A" else "Beta: N/A",
            "",
            "--- PROFITABILITY ---",
            f"Profit Margin: {profit_margin}",
            f"Operating Margin: {op_margin}",
            f"Return on Equity: {roe}",
            "",
            "--- FINANCIAL HEALTH ---",
            f"Debt/Equity: {de}",
            f"Current Ratio: {current_r}" if current_r != "N/A" else "Current Ratio: N/A",
        ]

        return "\n".join(lines)

    except Exception as e:
        return f"Error fetching fundamentals for {ticker}: {e}"


# ─────────────────────────────────────────────
# Tool 4: SEC Filings (10-K, 10-Q, 8-K)
# ─────────────────────────────────────────────

@tool("Fetch SEC Filings")
def fetch_sec_filings(ticker: str) -> str:
    """Fetches the latest 10-K, 10-Q, and 8-K filings from SEC EDGAR for any stock ticker. Returns form type, filing date, period end date, description, and SEC.gov link for each filing."""
    try:
        cik = _ticker_to_cik(ticker)
        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        resp = requests.get(url, headers=_SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        periods = recent.get("reportDate", [])
        descs = recent.get("primaryDocDescription", [])
        nums = recent.get("accessionNumber", [])

        target_forms = {"10-K", "10-Q", "8-K"}
        lines = [f"--- SEC FILINGS FOR {ticker.upper()} ---"]
        count = 0
        for i in range(len(forms)):
            if count >= 10:
                break
            if forms[i] in target_forms:
                acc = nums[i].replace("-", "")
                link = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{nums[i]}-index.htm"
                lines.append("")
                lines.append(f"Form: {forms[i]}")
                lines.append(f"Filed: {dates[i]}")
                lines.append(f"Period End: {periods[i] if i < len(periods) else 'N/A'}")
                lines.append(f"Description: {descs[i] if i < len(descs) and descs[i] else 'N/A'}")
                lines.append(f"Link: {link}")
                count += 1

        if count == 0:
            return f"No 10-K, 10-Q, or 8-K filings found for {ticker}."

        return "\n".join(lines)

    except Exception as e:
        return f"Error fetching SEC filings for {ticker}: {e}"


# ─────────────────────────────────────────────
# Tool 5: Evaluate Past Predictions
# ─────────────────────────────────────────────

@tool("Evaluate Past Predictions")
def evaluate_past_predictions(ticker: str) -> str:
    """Reads the notebook_memory history for a ticker and compares past predictions (entry, stop, targets, rating) against actual subsequent price movements. Returns a report of prediction accuracy."""
    filepath = os.path.join(_MEMORY_DIR, f"{ticker}_history.md")
    if not os.path.exists(filepath):
        return f"No history found for {ticker}."

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract analysis blocks: each starts with "## Analysis Date: YYYY-MM-DD"
    blocks = re.split(r'(?=## Analysis Date:)', content)
    comparisons = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        date_match = re.search(r"## Analysis Date:\s*(\d{4}-\d{2}-\d{2})", block)
        if not date_match:
            continue
        analysis_date_str = date_match.group(1)
        try:
            analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        extract = {}
        for label in ["Entry", "Stop Loss", "Target 1", "Target 2"]:
            m = re.search(rf"{label}:\s*\$?([\d,.]+)", block)
            if m:
                extract[label] = float(m.group(1).replace(",", ""))
        rating_m = re.search(r"Rating:\s*(🔴|🟡|🟢)?\s*(BUY|HOLD|SELL)", block)
        if rating_m:
            extract["Rating"] = rating_m.group(2)

        if "Entry" not in extract:
            continue

        # Fetch actual prices at +1w, +1m, +3m from analysis date
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=analysis_date_str, end=(datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d"))
            if hist.empty:
                continue

            actual_entry = hist['Close'].iloc[0] if len(hist) > 0 else None
            price_1w = hist['Close'].iloc[min(5, len(hist)-1)] if len(hist) > 5 else hist['Close'].iloc[-1]
            price_1m = hist['Close'].iloc[min(21, len(hist)-1)] if len(hist) > 21 else hist['Close'].iloc[-1]
            price_3m = hist['Close'].iloc[min(63, len(hist)-1)] if len(hist) > 63 else hist['Close'].iloc[-1]
            current_price = hist['Close'].iloc[-1]

            pnl_1w = ((price_1w - actual_entry) / actual_entry) * 100
            pnl_1m = ((price_1m - actual_entry) / actual_entry) * 100

            entered_stop = "Stop Hit" if extract.get("Stop Loss") and actual_entry and min(price_1w, price_1m) <= extract["Stop Loss"] else "Active"
            entered_target = f"Target 1 Hit" if extract.get("Target 1") and max(price_1w, price_1m, current_price) >= extract["Target 1"] else "Not Yet"

            comparisons.append(f"""
--- Analysis: {analysis_date_str} ---
Predicted Entry: ${extract.get('Entry', 'N/A'):.2f}
Actual Entry (open): ${actual_entry:.2f}
Predicted Stop: ${extract.get('Stop Loss', 'N/A')}
Predicted Targets: ${extract.get('Target 1', 'N/A')} / ${extract.get('Target 2', 'N/A')}
Rating: {extract.get('Rating', 'N/A')}
Actual Price +1w: ${price_1w:.2f} ({pnl_1w:+.2f}%)
Actual Price +1m: ${price_1m:.2f} ({pnl_1m:+.2f}%)
Actual Price +3m: ${price_3m:.2f}
Current Price: ${current_price:.2f}
Stop Status: {entered_stop}
Target Status: {entered_target}
Would you have profited? {"✅ YES" if pnl_1m > 0 else "❌ NO"}
""")
        except Exception:
            continue

    if not comparisons:
        return f"Could not evaluate predictions for {ticker}. History file may be empty or unparseable."

    # Summary statistics
    rating_accuracy = {"correct": 0, "total": 0}
    profitable = 0
    for c in comparisons:
        if "Would you have profited? ✅ YES" in c:
            profitable += 1

    header = f"""═══════════════════════════════════════
REFLECTION REPORT: {ticker.upper()}
Total Analyses Reviewed: {len(comparisons)}
Profitable Trades: {profitable}/{len(comparisons)} ({profitable/len(comparisons)*100:.0f}%)
═══════════════════════════════════════
"""
    return header + "\n".join(comparisons)
