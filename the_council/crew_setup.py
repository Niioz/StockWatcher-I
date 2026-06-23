import os
from datetime import datetime
from crewai import Agent, Task, Crew, Process, LLM
from tools import search_news, fetch_stock_analysis, fetch_fundamentals, fetch_sec_filings, evaluate_past_predictions
from summarizer import save_summary_report, archive_notebook_memory


# Model fallback chain — tries cheapest/most available first, escalates on quota errors
MODEL_CHAIN = [
    "gemini/gemini-3.1-flash-lite",  #  15 RPM | 500 RPD — most available, workhorse
    "gemini/gemini-3.5-flash",       #   5 RPM |  20 RPD — better quality
    "gemini/gemini-2.5-flash",       #   5 RPM |  20 RPD — stable fallback
]


def _run_crew_with_fallback(crew, memory_key: str):
    """Runs the crew, automatically retrying with lower models on 503/429."""
    api_key = os.getenv("GEMINI_API_KEY")
    for model_string in MODEL_CHAIN:
        # Create a fully resolved LLM instance for this model
        llm = LLM(model=model_string, api_key=api_key)
        for agent in crew.agents:
            agent.llm = llm
        try:
            print(f"[Council] Trying model: {model_string}")
            result = crew.kickoff()
            save_to_notebook_memory(memory_key, str(result))
            print(f"[Council] Success with model: {model_string}")
            return result
        except Exception as e:
            err_str = str(e)
            is_rate_error = any(code in err_str for code in [
                "503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "quota"
            ])
            if is_rate_error:
                print(f"[Council] {model_string} hit limit — trying next model...")
                continue
            raise  # Not a rate limit — fail immediately
    raise RuntimeError(
        "All models in the fallback chain are exhausted. "
        "Try again later or check your API quota."
    )


def _load_skill(filename: str) -> str:
    """Loads a skill backstory from the skills/ folder."""
    skills_dir = os.path.join(os.path.dirname(__file__), "skills")
    filepath = os.path.join(skills_dir, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Skill file {filename} not found. Use default behaviour."


def create_crew_for_ticker(ticker: str):
    """Creates and runs a CrewAI team for a specific ticker."""

    analysis_date = datetime.now().strftime("%B %d, %Y")

    # Start with the best model; _run_crew_with_fallback will downgrade if needed
    gemini_llm = MODEL_CHAIN[0]

    # --- AGENTS ---
    news_reader = Agent(
        role='News Analyst',
        goal=f'Find the 3-5 most important news items for {ticker} from the last 30 days. For each: provide exact headline, source, publication date, 1-2 sentence summary, explain WHY it moves the stock, and classify sentiment.',
        backstory=_load_skill("skills_news_reader.md"),
        verbose=True,
        allow_delegation=False,
        tools=[search_news],
        llm=gemini_llm
    )

    math_whiz = Agent(
        role='Technical & Fundamental Analyst',
        goal=f'Calculate all technical indicators (EMA20/50/200, RSI14, ATR, Bollinger Bands, support/resistance, trend) and fetch fundamental data (market cap, earnings date, analyst consensus, financial metrics) for {ticker}.',
        backstory=_load_skill("skills_math_whiz.md"),
        verbose=True,
        allow_delegation=False,
        tools=[fetch_stock_analysis, fetch_fundamentals],
        llm=gemini_llm
    )

    safety_guard = Agent(
        role='Risk Manager',
        goal=f'Identify the top 5 risks for {ticker} based on news sentiment, volatility data, valuation, and technical levels. Recommend stop-loss and position size.',
        backstory=_load_skill("skills_safety_guard.md"),
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm
    )

    the_boss = Agent(
        role='Senior Equity Research Analyst',
        goal=f'Synthesize all data into a professional-grade investment report for {ticker} following the exact format provided. Include bull case, bear case, trade setup, and final verdict.',
        backstory=_load_skill("skills_the_boss.md"),
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm
    )

    # --- TASKS ---
    task_news = Task(
        description=f'''
        ╔══════════════════════════════════════════════════════════════╗
        ║  STEP 1: Call your "Search Internet for News" tool with    ║
        ║          query = "{ticker}"                                ║
        ║  STEP 2: Read the tool's output carefully                  ║
        ║  STEP 3: Copy the exact headlines, sources, dates,         ║
        ║          and summaries into your response below            ║
        ║  DO NOT write "Data unavailable" — the tool IS available   ║
        ║  and WILL return real news for {ticker}.                   ║
        ╚══════════════════════════════════════════════════════════════╝

        Find the 3-5 most important recent news items for {ticker} using the search tool.
        ALL news must be from the last 30 days. If any item is older, explicitly label it:
        ⚠️ OUTDATED INFORMATION

        For EACH news item provide:

        📰 Headline:
        [exact headline from the tool output — copy it verbatim]

        🏢 Source:
        [publisher name from the tool output]

        📅 Published:
        [date from the tool output]

        💡 Summary:
        [1-2 sentence plain English summary]

        🔗 Why It Matters:
        [2-3 sentences explaining the cause-effect link to {ticker}'s stock price]

        Sentiment:
        🟢 Bullish / 🔴 Bearish / 🟡 Neutral
        ''',
        expected_output='''
        3-5 news items with real data (not "Data unavailable"). Each item has: Headline, Source, Published date, Summary, Why It Matters, Sentiment.
        ''',
        agent=news_reader
    )

    task_math = Task(
        description=f'''
        ╔══════════════════════════════════════════════════════════════╗
        ║  CRITICAL: You MUST call BOTH "Fetch Technical Analysis"   ║
        ║  AND "Fetch Fundamental Data" tools. Do NOT write any      ║
        ║  numbers from memory. Only report what the tools return.   ║
        ╚══════════════════════════════════════════════════════════════╝

        Use the Technical Analysis tool and Fundamental Data tool for {ticker}.
        Report ALL of the following:

        ── TECHNICAL ──
        Current Price, Trend (uptrend/downtrend), EMA20, EMA50, EMA200,
        Support levels (20d, 50d), Resistance levels (20d, 50d),
        RSI 14, ATR 14, Bollinger Bands (upper/middle/lower/width),
        Daily Return Mean, Std Dev (Volatility), Sharpe Ratio (1y),
        Fibonacci Retracement (90-day swing high/low + levels),
        Pivot Points (classic: PP, R1/R2/R3, S1/S2/S3),
        60-Day ROI, 1-Year ROI,
        Latest Volume vs 20-day Average.

        ── FUNDAMENTAL ──
        Company Name, Sector, Industry, Market Cap,
        Next Earnings Date, Analyst Consensus, Mean Target Price,
        Target Price Range (low-high),
        Total Revenue, Revenue Growth (YoY), EPS (TTM), EPS (Forward),
        Free Cash Flow, P/E (TTM), P/E (Forward), P/S, PEG Ratio, Beta,
        Profit Margin, Operating Margin, ROE, Debt/Equity, Current Ratio.

        If any data is unavailable, state "Data unavailable" — do not invent numbers.
        ''',
        expected_output='''
        A complete technical and fundamental data dump for {ticker} covering all indicators, quant calculations, and financial metrics listed above.
        ''',
        agent=math_whiz
    )

    task_risk = Task(
        description=f'''
        ╔══════════════════════════════════════════════════════════════╗
        ║  CRITICAL: Base your risk analysis ONLY on the data        ║
        ║  provided by the News Analyst and Technical Analyst.       ║
        ║  Do NOT invent risk factors from your own knowledge.       ║
        ╚══════════════════════════════════════════════════════════════╝

        Based on the News Analyst's sentiment and the Technical/Fundamental Analyst's data for {ticker},
        identify the TOP 5 RISKS for this stock. Consider:

        - News-driven risks (regulatory, macro, competitive)
        - Technical risks (overbought RSI, breakdown below support, high volatility)
        - Valuation risks (high P/E, PEG above 3, low margins)
        - Sector/macro risks (interest rates, commodity prices, geopolitical)
        - Company-specific risks (upcoming earnings, debt levels, concentration)

        For each risk: name it, explain it in one sentence, and rate severity (HIGH / MEDIUM / LOW).

        Then recommend:
        - Exact Stop-Loss price
        - Suggested Position Size (1%, 2%, 5%, or 10% of portfolio)
        ''',
        expected_output='''
        Top 5 risks for {ticker} with severity ratings, plus stop-loss and position size recommendation.
        ''',
        agent=safety_guard
    )

    task_final_report = Task(
        description=f'''
        ╔══════════════════════════════════════════════════════════════╗
        ║  CRITICAL: Today is {analysis_date}.                       ║
        ║                                                           ║
        ║  Three reports are provided below (News → Technical →     ║
        ║  Risk). You MUST extract EVERY number from those reports. ║
        ║  DO NOT write any price, date, percentage, or headline    ║
        ║  from your own knowledge. If it's not in the reports      ║
        ║  below, write "N/A".                                      ║
        ║                                                           ║
        ║  DATA AUDIT: After writing, re-read your report. For      ║
        ║  every number, ask: "Did this come from the reports       ║
        ║  above, or from my training data?" If from training,      ║
        ║  replace it with "N/A".                                   ║
        ╚══════════════════════════════════════════════════════════════╝

        You are a Senior Equity Research Analyst. Synthesize all the reports from the News Analyst, Technical & Fundamental Analyst, and Risk Manager into a professional-grade investment report for {ticker}.

        MANDATORY RULES:
        1. Use current data only from the reports provided.
        2. If any data appears older than 30 days, label it: ⚠️ OUTDATED INFORMATION
        3. Do not invent numbers. If unavailable, state "Data unavailable."
        4. Include both bull and bear cases.

        OUTPUT FORMAT — Use EXACTLY:

        ━━━━━━━━ STOCK ANALYSIS ━━━━━━━━

        Ticker: {ticker}
        Company: [company name]
        Analysis Date: {analysis_date}
        Current Price: $[price]
        Market Cap: [market cap]

        ━━━━━━━━ NEWS CATALYSTS ━━━━━━━━

        [For each of the 3-5 news items, provide:]

        📰 Headline:
        [headline]

        🏢 Source:
        [source]

        📅 Published:
        [date]

        💡 Summary:
        [1-2 sentence summary]

        🔗 Why It Matters:
        [why it moves {ticker}'s price]

        Sentiment:
        🟢 Bullish / 🔴 Bearish / 🟡 Neutral

        ━━━━━━━━ TECHNICAL ANALYSIS ━━━━━━━━

        Current Trend: [Uptrend / Downtrend / Sideways]
        Support Levels: [20d: $X | 50d: $X]
        Resistance Levels: [20d: $X | 50d: $X]

        EMA20: $X (Price above/below)
        EMA50: $X (Price above/below)
        EMA200: $X (Price above/below)

        RSI 14: [value] ([overbought/oversold/neutral])
        ATR 14: $[value]

        Bollinger Bands:
        Upper: $X | Middle: $X | Lower: $X | Width: X%

        Sharpe Ratio (1y): [value]

        Fibonacci Retracement (90d):
        Swing High: $X | Swing Low: $X
        Upside: 23.6%: $X | 38.2%: $X | 50%: $X | 61.8%: $X | 78.6%: $X
        Downside: 23.6%: $X | 38.2%: $X | 50%: $X | 61.8%: $X | 78.6%: $X

        Pivot Points (classic):
        PP: $X | R1: $X | R2: $X | R3: $X | S1: $X | S2: $X | S3: $X

        Technical Rating:
        [Strong Buy / Buy / Neutral / Sell / Strong Sell]

        ━━━━━━━━ FUNDAMENTAL ANALYSIS ━━━━━━━━

        Revenue (TTM): [value]
        Revenue Growth: [%]
        EPS (TTM): [value]
        EPS (Forward): [value]
        Free Cash Flow: [value]

        Valuation:
        P/E (TTM): [value]
        P/E (Forward): [value]
        P/S: [value]
        PEG: [value]
        Beta: [value]

        Profitability:
        Profit Margin: [%]
        Operating Margin: [%]
        ROE: [%]

        Financial Health:
        Debt/Equity: [%]
        Current Ratio: [value]

        ━━━━━━━━ SEC FILINGS ━━━━━━━━

        Update `stocks.md` with the latest 10-K, 10-Q, and 8-K filings.
        If SEC filings data is unavailable, state "Fetch with !filings command."

        ━━━━━━━━ RISK ANALYSIS ━━━━━━━━

        Top 5 Risks:

        1. [Risk Name] — [Severity: HIGH/MEDIUM/LOW]
           [one sentence explanation]

        2. ...

        3. ...

        4. ...

        5. ...

        ━━━━━━━━ BULL CASE vs BEAR CASE ━━━━━━━━

        🟢 Bull Case:
        [2-3 sentence argument for the upside]

        🔴 Bear Case:
        [2-3 sentence argument for the downside]

        ━━━━━━━━ TRADE SETUP ━━━━━━━━

        Entry: $[price]
        Stop Loss: $[price]
        Target 1: $[price]
        Target 2: $[price]

        Risk/Reward Ratio: [X]:1
        Suggested Position Size: [1% / 2% / 5% / 10%]

        Confidence Score: [0-100]/100

        ━━━━━━━━ FINAL VERDICT ━━━━━━━━

        Rating: 🟢 BUY / 🟡 HOLD / 🔴 SELL

        Investment Horizon:
        Short-Term: [weeks]
        Medium-Term: [months]
        Long-Term: [years]

        One-Sentence Thesis:
        [one clear sentence]

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ''',
        expected_output='''
        A complete, professional-grade investment report for {ticker} following the exact format with all sections populated.
        ''',
        agent=the_boss
    )

    # --- CREW ---
    crew = Crew(
        agents=[news_reader, math_whiz, safety_guard, the_boss],
        tasks=[task_news, task_math, task_risk, task_final_report],
        process=Process.sequential,
        verbose=True
    )

    result = _run_crew_with_fallback(crew, ticker)
    return result

def save_to_notebook_memory(ticker: str, content: str):
    """Saves the report to a local folder so NotebookLM can ingest it, and the bots can learn."""
    memory_dir = "notebook_memory"
    os.makedirs(memory_dir, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = os.path.join(memory_dir, f"{ticker}_history.md")
    
    # Append to the history file so the bots have a running log to learn from
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n\n## Analysis Date: {date_str}\n")
        f.write(content)
        f.write("\n---\n")

def create_crew_for_reflect(period: str = None):
    """Creates and runs a CrewAI team that reviews past predictions and generates a reflection report.
    If period is 'weekly', 'monthly', or 'quarterly', filters entries by date range and generates a summary."""

    analysis_date = datetime.now().strftime("%B %d, %Y")
    gemini_llm = MODEL_CHAIN[0]

    # Scan notebook_memory for ticker history files
    memory_dir = "notebook_memory"
    tickers = []
    if os.path.exists(memory_dir):
        for f in os.listdir(memory_dir):
            if f.endswith("_history.md"):
                ticker = f.replace("_history.md", "")
                tickers.append(ticker)

    if not tickers:
        return "No past analyses found in notebook_memory. Run !analyze first."

    historian = Agent(
        role='Historian',
        goal='Review past analysis predictions and compare them against actual price movements to measure accuracy.',
        backstory='''You are the firm's Historian — part auditor, part quant. Your job is to dig through past research reports,
        extract every prediction (entry price, stop-loss, target prices, rating), fetch real market data for the subsequent period,
        and produce a cold, hard accuracy scorecard. You do not sugarcoat. You do not make excuses for the models.
        If the Council was wrong, you say so plainly.''',
        verbose=True,
        allow_delegation=False,
        tools=[evaluate_past_predictions],
        llm=gemini_llm
    )

    strategist = Agent(
        role='Senior Strategist',
        goal='Synthesize the Historian findings into an actionable reflection report with lessons and process improvements.',
        backstory='''You are the Senior Strategist at a top-tier family office. You review the Historian's accuracy report
        and translate it into clear lessons. What worked? What didn't? Should we adjust our models, risk parameters, or
        research process? Your report is concise, data-driven, and actionable for the CIO.''',
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm
    )

    # Create tasks for each ticker
    ticker_tasks = []
    for ticker in tickers:
        task = Task(
            description=f'''
            Call the "Evaluate Past Predictions" tool with ticker = "{ticker}".
            Read the tool output carefully. Copy the complete reflection data for {ticker} into your response.
            Include: number of analyses, profitable trades, each analysis comparison with prices and P&L.
            ''',
            expected_output=f'Complete prediction accuracy data for {ticker} from the evaluate tool.',
            agent=historian
        )
        ticker_tasks.append(task)

    # Synthesizer task
    period_label = "All-Time"
    period_desc = "all analyses in the database"
    if period == "weekly":
        period_label = "Weekly"
        period_desc = "analyses from the past 7 days"
    elif period == "monthly":
        period_label = "Monthly"
        period_desc = "analyses from the past 30 days"
    elif period == "quarterly":
        period_label = "Quarterly"
        period_desc = "analyses from the past 90 days"

    synthesizer_task = Task(
        description=f'''
        ╔══════════════════════════════════════════════════════════════╗
        ║  Today: {analysis_date}                                    ║
        ║  Report Type: {period_label} Reflection                    ║
        ║                                                           ║
        ║  You have the Historian's per-ticker accuracy data below. ║
        ║  Synthesize it into a clear reflection report.            ║
        ║  Do NOT invent numbers. Only use data from reports.       ║
        ╚══════════════════════════════════════════════════════════════╝

        Write a {period_label} Reflection Report covering {period_desc}.

        FORMAT:
        ━━━【 {period_label} REFLECTION REPORT 】━━━
        Date: {analysis_date}
        Period: {period_desc}
        Tickers Reviewed: {', '.join(tickers)}

        ━━━ ACCURACY SCORECARD ━━━
        [For each ticker:]
        {ticker}:
        - Analyses Reviewed: [N]
        - Profitable Trades: [N/N] ([X]%)
        - Most Accurate Prediction: [which rating/timeframe]
        - Least Accurate Prediction: [which rating/timeframe]
        - Key Lesson: [1 sentence]

        ━━━ TOP PROCESS IMPROVEMENTS ━━━
        1. [Actionable improvement based on data, e.g., "Stop-losses were too tight on high-volatility names"]
        2. ...
        3. ...

        ━━━ POSITION-SPECIFIC NOTES ━━━
        [Any ticker-specific warnings or adjustments]

        ━━━ [End of Report] ━━━
        ''',
        expected_output=f'A complete {period_label} reflection report with per-ticker accuracy, win rates, and process improvements.',
        agent=strategist
    )

    crew = Crew(
        agents=[historian, strategist],
        tasks=ticker_tasks + [synthesizer_task],
        process=Process.sequential,
        verbose=True
    )

    result = _run_crew_with_fallback(crew, "REFLECTION")

    # After reflect runs, save summary and archive if periodic
    if period in ("weekly", "monthly", "quarterly"):
        save_summary_report(period)
        archive_notebook_memory(period)

    return result


def create_crew_for_portfolio():
    """Runs a CrewAI task to review the user's specific portfolio."""
    analysis_date = datetime.now().strftime("%B %d, %Y")
    gemini_llm = MODEL_CHAIN[0]
    
    portfolio_manager = Agent(
        role='Portfolio Manager',
        goal='Analyze the current portfolio health using real-time data from the tools. Do NOT use your training memory for prices or valuations.',
        backstory=_load_skill("skills_the_boss.md"),
        verbose=True,
        allow_delegation=False,
        tools=[fetch_stock_analysis, fetch_fundamentals],
        llm=gemini_llm
    )
    
    task_portfolio = Task(
        description=f'''
        ╔══════════════════════════════════════════════════════════════╗
        ║  CRITICAL: You MUST call the tools to get current prices.  ║
        ║  Do NOT write any price, percentage, or valuation from     ║
        ║  your training memory. Only use what the tools return.     ║
        ╚══════════════════════════════════════════════════════════════╝

        Analysis Date: {analysis_date}

        Analyze the client's current portfolio:
        - NASDAQ-A: 30.74%
        - CCJ: 27.93%
        - OKLO: 22.54%
        - MSFT: 16.15%
        - META: 2.64%
        
        Use the Technical Analysis and Fundamental Data tools to get current data on CCJ, OKLO, MSFT, and META.
        Write a Discord report advising if they are over-exposed to nuclear (CCJ, OKLO) and what to do with the tiny META position.
        Format it clearly with emojis and action items for each holding.
        ''',
        expected_output="A formatted Discord message with a health check and action items for the portfolio.",
        agent=portfolio_manager
    )
    
    crew = Crew(
        agents=[portfolio_manager],
        tasks=[task_portfolio],
        process=Process.sequential,
        verbose=True
    )
    
    result = _run_crew_with_fallback(crew, "PORTFOLIO")
    return result
