# The Council — AGENTS.md

## Project structure
- All code lives under `the_council/`
- No README, no test framework, no linter, no formatter config

## Setup & run
- `start_bot.bat` creates venv, installs deps, runs `bot.py`
- Python 3.12 hardcoded in the bat file
- `.env` must contain `DISCORD_TOKEN`, `GEMINI_API_KEY`, `SERPER_API_KEY`
- **Never commit real API keys** — `.gitignore` excludes `.env`, `venv/`, `__pycache__/`
- After pulling, run `pip install -r requirements.txt` to get new deps

## Architecture
- **`bot.py`** — Discord bot entrypoint (prefix `!`). Commands: `!analyze`, `!portfolio`, `!reflect`, `!filings`, `!help`
- **`crew_setup.py`** — CrewAI agent orchestration. Imports `setup_tracing()` on load (see tracing section)
- **`tools.py`** — 5 CrewAI `@tool` functions (news, technicals, fundamentals, SEC filings, prediction eval)
- **`summarizer.py`** — Period-based summary & archive for notebook memory


## CrewAI specifics
- Model fallback chain (in order): `gemini-3.1-flash-lite` → `gemini-3.5-flash` → `gemini-2.5-flash`
- All agents use `Process.sequential`
- Skill backstories are in `skills/*.md` loaded by `_load_skill()`
- Portfolio tickers hardcoded in `bot.py:102` and `crew_setup.py:561-565` — change in both places
- Notebook memory saved to `notebook_memory/{TICKER}_history.md`
- SEC filings saved to `stocks.md`

## News search fallback chain
1. Yahoo Finance (`yf.Ticker.news`) — no key needed
2. Serper API (`google.serper.dev/news`) — requires `SERPER_API_KEY`
3. DuckDuckGo (`ddgs.news()`) — no key needed, free fallback

## Tracing
- Built-in CrewAI tracing: uncomment `CREWAI_TRACING_ENABLED=true` in `.env`
