"""
live/scheduler.py
-----------------
Runs the daily briefing automatically every weekday at 9:00 AM.
Keep this running in the background on your computer.

Usage:
    python live/scheduler.py

To run in background on Windows:
    pythonw live/scheduler.py
"""

import schedule
import time
from datetime import datetime
from loguru import logger
import sys
sys.path.insert(0, ".")


def job():
    """Runs the daily briefing."""
    now = datetime.now()
    # Skip weekends
    if now.weekday() >= 5:
        logger.info("Weekend — skipping")
        return
    logger.info(f"Running daily briefing at {now.strftime('%H:%M')}...")
    try:
        from live.daily import run_daily
        run_daily()
    except Exception as e:
        logger.error(f"Daily run failed: {e}")


# Schedule for 9:00 AM every day
schedule.every().day.at("09:00").do(job)

if __name__ == "__main__":
    logger.info("Scheduler started. Will run daily at 9:00 AM on weekdays.")
    logger.info("Press Ctrl+C to stop.")

    # Run immediately on startup so you can test it
    logger.info("Running once now to test...")
    job()

    while True:
        schedule.run_pending()
        time.sleep(60)
