# StockWatcher I — The Council

**AI-powered stock research, delivered through Discord.**

StockWatcher I brings a team of AI research agents into your Discord server. Ask **The Council** to analyze a stock, review a configured portfolio, look up SEC filings, or revisit earlier predictions using later market data.

Built with **Python**, **CrewAI**, **Google Gemini**, **discord.py**, and **yfinance**. Reports are saved locally as Markdown so you can keep a research history and review how the analysis changes over time.

## Features

- **Stock research:** Combine recent news, technical indicators, company fundamentals, and risk analysis in one report.
- **Technical analysis:** Calculate EMA 20/50/200, RSI, ATR, Bollinger Bands, support and resistance, Fibonacci retracements, pivot points, volatility, returns, and a Sharpe ratio.
- **Fundamental analysis:** Retrieve revenue, earnings, cash flow, valuation ratios, margins, debt metrics, and analyst targets where available.
- **Portfolio reviews:** Review the holdings and allocations configured in the project, with concentration concerns and per-holding action items.
- **SEC filing lookups:** Fetch up to 10 recent 10-K, 10-Q, and 8-K filing entries and save their metadata and links to `stocks.md`.
- **Prediction reviews:** Compare saved research with subsequent prices and request weekly, monthly, or quarterly reflections.
- **Local research history:** Keep analysis reports, period summaries, and archived entries in Markdown files.

## How it works

For `!analyze`, CrewAI runs four agents sequentially:

| Agent | Responsibility |
| --- | --- |
| News Analyst | Gather headlines, publication dates, sources, summaries, and sentiment. |
| Technical & Fundamental Analyst | Fetch market data and calculate indicators and financial metrics. |
| Risk Manager | Assess the research, identify risks, and propose stop-loss and position-size levels. |
| Senior Equity Research Analyst | Combine the findings into a report with bull and bear cases, a trade setup, and a final thesis. |

News retrieval tries **Yahoo Finance → Serper → DuckDuckGo**, moving to the next source if no results are returned. Gemini model selection uses a configurable fallback list and retries on recognized quota or temporary availability errors.

## Getting started

### Prerequisites

- **Python 3.12**, matching the version used by the included Windows launcher.
- **Git** to clone the repository.
- A Discord bot token from the [Discord Developer Portal](https://discord.com/developers/applications).
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
- Optionally, a [Serper API key](https://serper.dev/) for the Google News fallback.

### 1. Configure your Discord bot

Create a Discord application and bot, then copy its bot token for the environment setup below.

On the application's **Bot** page, enable **Message Content Intent** under **Privileged Gateway Intents**. The Council uses `!` prefix commands and needs access to message content. See [Discord's intent documentation](https://docs.discord.com/developers/events/gateway#enabling-privileged-intents).

Invite the bot to your server with the `bot` OAuth2 scope and channel permissions to **View Channels** and **Send Messages**.

### 2. Clone and install

```bash
git clone https://github.com/Niioz/StockWatcher-I.git
cd StockWatcher-I/the_council
```

**Windows — PowerShell**

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
python3.12 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

These commands use the virtual environment directly, so activation is optional.

### 3. Add your credentials

Edit `the_council/.env` and replace the placeholder values:

```dotenv
DISCORD_TOKEN=your_discord_bot_token_here
GEMINI_API_KEY=your_gemini_api_key_here
SERPER_API_KEY=

# Optional CrewAI tracing setting from .env.example:
# CREWAI_TRACING_ENABLED=true
```

| Variable | Required | Purpose |
| --- | --- | --- |
| `DISCORD_TOKEN` | Yes | Connect the bot to Discord. |
| `GEMINI_API_KEY` | Yes, for AI reports | Run the Gemini-backed research agents. |
| `SERPER_API_KEY` | No | Enable the Serper news fallback. Leave blank to skip it. |

Keep real credentials in `.env`, which is excluded by the project's `.gitignore`.

### 4. Start The Council

Run from **`the_council/`** so report history is written to the expected location.

**Windows — PowerShell**

```powershell
.\venv\Scripts\python.exe bot.py
```

**macOS / Linux**

```bash
./venv/bin/python bot.py
```

Once the bot reports that it is online, send `!help` in a channel it can access.

**Windows launcher:** You can also use `start_bot.bat`, which creates a virtual environment if needed, installs dependencies, and starts the bot. Update its `PYTHON_PATH` first: the checked-in script points to a machine-specific Python installation.

## Discord commands

| Command | What it does |
| --- | --- |
| `!analyze MSFT` | Research a ticker and return a stock analysis report. |
| `!portfolio` | Review the portfolio configured in `crew_setup.py`. |
| `!reflect` | Review predictions in the retained research history. |
| `!reflect weekly` | Request a weekly reflection, summarize the past 7 days, and archive older entries. |
| `!reflect monthly` | Request a monthly reflection, summarize the past 30 days, and archive older entries. |
| `!reflect quarterly` | Request a quarterly reflection, summarize the past 90 days, and archive older entries. |
| `!filings MSFT` | Retrieve recent SEC filing entries and update `stocks.md`. |
| `!help` | Show the command manual. |

Replace `MSFT` with the ticker you want to research. Long reports are split across Discord messages. Run `!filings` separately when you want to update the filing database.

## Research history and reflections

Analysis and portfolio reports are appended to `notebook_memory/{TICKER}_history.md`; portfolio and reflection reports use the keys `PORTFOLIO` and `REFLECTION`. The repository already includes saved research examples.

Periodic reflections generate summaries under `summaries/` when matching entries exist and move entries older than the selected window into `summaries/archive/`. **Archiving updates the active history files**, so later reviews read the retained entries. Markdown history can also be imported manually into tools such as NotebookLM.

Reflection scores are approximate comparisons of selected daily closing prices. If a forecast horizon has not elapsed, the evaluator uses the latest available close. These results do not simulate complete trade execution or trading costs. The period argument guides the reflection prompt and controls summary and archive cutoffs; the price evaluator itself reads all retained entries for each ticker.

## Customization

| Setting | Where to change it |
| --- | --- |
| Portfolio holdings and allocations | Edit `create_crew_for_portfolio()` in [crew_setup.py](the_council/crew_setup.py), including its tool instructions. Update the portfolio status and help text in [bot.py](the_council/bot.py) to match. |
| Gemini models | Edit `MODEL_CHAIN` in [crew_setup.py](the_council/crew_setup.py) to use model IDs available to your API key. |
| Agent behavior | Edit the Markdown backstories in [skills/](the_council/skills/). |
| Report format and task instructions | Edit the task definitions in [crew_setup.py](the_council/crew_setup.py). |
| SEC request identification | Replace the placeholder contact in `_SEC_HEADERS` in [tools.py](the_council/tools.py) with your application name and contact email, following [SEC access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data). |

The portfolio is currently hardcoded; customize it before using `!portfolio` for your own holdings.

## Project structure

```text
StockWatcher-I/
├── AGENTS.md                  # Project guidance for coding agents
└── the_council/
    ├── .env.example            # Environment variable template
    ├── .gitignore
    ├── bot.py                  # Discord commands and report delivery
    ├── crew_setup.py           # Agents, tasks, model fallback, report history
    ├── tools.py                # News, market data, filings, prediction evaluation
    ├── summarizer.py           # Period summaries and history archiving
    ├── requirements.txt        # Python dependencies
    ├── start_bot.bat           # Windows launcher
    ├── skills/                # Agent backstories
    ├── notebook_memory/       # Saved research reports
    ├── stocks.md              # SEC filing metadata and links
    └── summaries/             # Generated period summaries and archives
```

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Bot is offline or exits at startup | Set a valid `DISCORD_TOKEN` in `.env` and confirm your environment has the required packages. |
| Bot does not respond to commands | Enable Message Content Intent, check channel permissions, and use the `!` prefix. |
| Gemini authentication, quota, or model errors | Check your key, provider quota, and `MODEL_CHAIN`. Recognized quota and availability errors trigger fallback; other errors stop the run. |
| Missing market data or news | Check the ticker and provider availability. Some symbols have incomplete history or financial metrics. |
| SEC lookups fail | Check the ticker's SEC coverage, your `_SEC_HEADERS` contact information, and SEC access restrictions. |
| History appears in the wrong folder | Start the bot from `the_council/`. Some history paths are relative to the working directory. |
| Windows launcher cannot find Python | Update `PYTHON_PATH` in `start_bot.bat` or use the manual launch commands above. |

## Contributing

Bug reports and suggestions are welcome through [GitHub Issues](https://github.com/Niioz/StockWatcher-I/issues). Include the command, ticker, Python version, and relevant error details when reporting a problem. Remove tokens and API keys from logs before sharing them.

## License

This repository does not currently include a `LICENSE` file.

## Research notice

The Council produces AI-generated research for educational and informational use. Verify its data, calculations, and conclusions independently before making investment decisions. Market data may be delayed or incomplete, and AI-generated ratings and trade setups can be wrong.
