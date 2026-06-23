import os
import re
import shutil
from datetime import datetime, timedelta, timezone

_MEMORY_DIR = os.path.join(os.path.dirname(__file__), "notebook_memory")
_SUMMARY_DIR = os.path.join(os.path.dirname(__file__), "summaries")


def _ensure_dirs():
    os.makedirs(_SUMMARY_DIR, exist_ok=True)
    os.makedirs(_MEMORY_DIR, exist_ok=True)


def _get_period_dates(period: str):
    """Returns (cutoff_date, period_label) based on period type."""
    now = datetime.now(timezone.utc)
    if period == "weekly":
        iso = now.isocalendar()
        label = f"weekly_{iso[0]}-W{iso[1]:02d}"
        cutoff = now - timedelta(days=7)
    elif period == "monthly":
        label = f"monthly_{now.strftime('%Y-%m')}"
        cutoff = now - timedelta(days=30)
    elif period == "quarterly":
        quarter = (now.month - 1) // 3 + 1
        label = f"quarterly_{now.year}-Q{quarter}"
        cutoff = now - timedelta(days=90)
    else:
        raise ValueError(f"Unknown period: {period}")
    return cutoff, label


def save_summary_report(period: str):
    """Scans notebook_memory, filters entries by period cutoff, writes a summary.md."""
    _ensure_dirs()
    cutoff, label = _get_period_dates(period)

    ticker_data = {}
    if not os.path.exists(_MEMORY_DIR):
        return

    for filename in os.listdir(_MEMORY_DIR):
        if not filename.endswith("_history.md"):
            continue
        ticker = filename.replace("_history.md", "")
        filepath = os.path.join(_MEMORY_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = re.split(r'(?=## Analysis Date:)', content)
        period_entries = []
        for block in blocks:
            date_match = re.search(r"## Analysis Date:\s*(\d{4}-\d{2}-\d{2})", block)
            if not date_match:
                continue
            try:
                entry_date = datetime.strptime(date_match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if entry_date >= cutoff:
                period_entries.append(block.strip())

        if period_entries:
            ticker_data[ticker] = period_entries

    if not ticker_data:
        return

    summary_lines = [
        f"# {label.upper()} SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        f"Period: {cutoff.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
        f"Tickers Analyzed: {', '.join(sorted(ticker_data.keys()))}",
        "",
        "---",
        ""
    ]

    for ticker in sorted(ticker_data.keys()):
        entries = ticker_data[ticker]
        summary_lines.append(f"## {ticker}")
        summary_lines.append(f"Entries in this period: {len(entries)}")
        summary_lines.append("")

        ratings = []
        entries_for_pnl = []
        for entry in entries:
            rating_m = re.search(r"Rating:\s*[🔴🟡🟢]?\s*(BUY|HOLD|SELL)", entry)
            if rating_m:
                ratings.append(rating_m.group(1))
            price_m = re.search(r"Current Price:\s*\$?([\d,.]+)", entry)
            if price_m:
                entries_for_pnl.append(float(price_m.group(1).replace(",", "")))

        if ratings:
            buys = ratings.count("BUY")
            holds = ratings.count("HOLD")
            sells = ratings.count("SELL")
            summary_lines.append(f"Ratings: {buys}x BUY | {holds}x HOLD | {sells}x SELL")

        if entries_for_pnl and len(entries_for_pnl) > 1:
            avg_price = sum(entries_for_pnl) / len(entries_for_pnl)
            summary_lines.append(f"Avg Analysis Price: ${avg_price:.2f} (across {len(entries_for_pnl)} entries)")
            summary_lines.append(f"Price Range: ${min(entries_for_pnl):.2f} — ${max(entries_for_pnl):.2f}")

        summary_lines.append("")
        for entry in entries:
            date_match = re.search(r"## Analysis Date:\s*(\d{4}-\d{2}-\d{2})", entry)
            date_str = date_match.group(1) if date_match else "unknown"
            rating_m = re.search(r"Rating:\s*[🔴🟡🟢]?\s*(BUY|HOLD|SELL)", entry)
            rating_str = rating_m.group(1) if rating_m else "N/A"
            price_m = re.search(r"Current Price:\s*\$?([\d,.]+)", entry)
            price_str = f"${price_m.group(1)}" if price_m else "N/A"
            summary_lines.append(f"- **{date_str}** | Rating: {rating_str} | Price: {price_str}")

        summary_lines.append("")
        summary_lines.append("---")
        summary_lines.append("")

    summary_path = os.path.join(_SUMMARY_DIR, f"{label}.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"[Summarizer] Saved {period} summary to {summary_path}")


def archive_notebook_memory(period: str):
    """Moves archived entries from notebook_memory into the summaries/archive folder."""
    _ensure_dirs()
    cutoff, label = _get_period_dates(period)
    archive_dir = os.path.join(_SUMMARY_DIR, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    if not os.path.exists(_MEMORY_DIR):
        return

    for filename in os.listdir(_MEMORY_DIR):
        if not filename.endswith("_history.md"):
            continue
        filepath = os.path.join(_MEMORY_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = re.split(r'(?=## Analysis Date:)', content)
        retained = []
        archived_entries = []
        for block in blocks:
            date_match = re.search(r"## Analysis Date:\s*(\d{4}-\d{2}-\d{2})", block)
            if not date_match:
                retained.append(block)
                continue
            try:
                entry_date = datetime.strptime(date_match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                retained.append(block)
                continue
            if entry_date >= cutoff:
                retained.append(block)
            else:
                archived_entries.append(block)

        # Write back retained entries
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n\n".join(retained))

        # Write archived entries to archive file
        if archived_entries:
            archive_path = os.path.join(archive_dir, f"{label}_{filename}")
            with open(archive_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n## Archived on: {datetime.now().strftime('%Y-%m-%d')}\n")
                f.write("\n\n".join(archived_entries))
                f.write("\n---\n")

    print(f"[Summarizer] Archived {period}-old entries to {archive_dir}")
