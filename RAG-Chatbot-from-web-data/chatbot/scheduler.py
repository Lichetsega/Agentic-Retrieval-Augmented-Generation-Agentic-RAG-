"""
Automated Web Crawling Scheduler for Visit Ethiopia RAG Chatbot.

This script manages automated web crawling and vector database re-indexing,
targeting Ethiopian Midnight (00:00:00 East Africa Time / UTC+3) or any specified daily target time.
It can run as a standalone CLI process or as a background thread inside api_server.py.

Usage:
    python scheduler.py                                       # Midnight schedule (00:00), immediate crawl
    python scheduler.py --target-time 09:20                   # Test schedule targeting 09:20 local time
    python scheduler.py --skip-initial --target-time 09:20   # Sleeps until 09:20 without initial crawl
    python scheduler.py --mode interval --interval-hours 24   # Fixed interval schedule
"""

import argparse
import datetime
import os
import sys
import time
import threading
import traceback
from dotenv import load_dotenv

# Ensure local module imports work seamlessly regardless of the directory from which the script is launched
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from core_utils import DEFAULT_URLS
from ingest import run_ingestion

# Explicit Ethiopian Timezone (East Africa Time: UTC+3)
try:
    from zoneinfo import ZoneInfo
    ETHIOPIA_TZ = ZoneInfo("Africa/Addis_Ababa")
except Exception:
    # Fallback if zoneinfo tzdata is unavailable on minimal environments
    ETHIOPIA_TZ = datetime.timezone(datetime.timedelta(hours=3), name="EAT")


def get_ethiopia_now() -> datetime.datetime:
    """Returns current date and time in Ethiopian Timezone (EAT / UTC+3)."""
    return datetime.datetime.now(ETHIOPIA_TZ)


def get_seconds_until_next_target_time(
    target_hour: int = 0,
    target_minute: int = 0,
    tz: datetime.tzinfo = ETHIOPIA_TZ,
) -> tuple[float, datetime.datetime]:
    """
    Calculates exact seconds remaining until the next targeted clock time (HH:MM:00 EAT).
    Defaults to 00:00 (Ethiopian Midnight).

    Returns:
        tuple[float, datetime.datetime]: (seconds_remaining, next_target_datetime)
    """
    now = datetime.datetime.now(tz)
    target = now.replace(
        hour=target_hour, minute=target_minute, second=0, microsecond=0
    )
    if target <= now:
        target += datetime.timedelta(days=1)
    seconds_remaining = (target - now).total_seconds()
    return seconds_remaining, target


def execute_scheduled_crawl(urls: list[str]) -> bool:
    """
    Executes a single web crawl and vector indexing cycle.

    Args:
        urls (list[str]): List of target website URLs to crawl and process.

    Returns:
        bool: True if the ingestion job ran without unhandled exceptions, False otherwise.
    """
    now_str = get_ethiopia_now().strftime("%Y-%m-%d %H:%M:%S %Z")

    print("\n" + "=" * 60)
    print(f"🚀 [SCHEDULER] Starting automated crawl cycle at {now_str}")
    print(f"🌐 Target sources: {len(urls)} URLs")
    print("=" * 60)

    try:
        success, failed = run_ingestion(urls)
        finish_str = get_ethiopia_now().strftime("%Y-%m-%d %H:%M:%S %Z")

        print("\n" + "-" * 60)
        print(f"✅ [SCHEDULER] Crawl completed at {finish_str}")
        print(f"📊 Results: {success} succeeded, {failed} failed")
        print("-" * 60 + "\n")
        return True

    except Exception as e:
        print(f"❌ [SCHEDULER ERROR] Unhandled exception during crawl: {e}")
        traceback.print_exc()
        return False


def start_scheduler_loop(
    mode: str = "midnight",
    target_hour: int = 0,
    target_minute: int = 0,
    interval_seconds: float = 86400.0,
    run_immediately: bool = False,
):
    """
    Main infinite scheduler loop that manages timing, initial executions, and sleep cycles.

    Args:
        mode (str): "midnight" (or "target") for daily fixed time target, or "interval" for fixed interval sleeping.
        target_hour (int): Target hour (0-23) for daily target schedule.
        target_minute (int): Target minute (0-59) for daily target schedule.
        interval_seconds (float): Duration in seconds between runs when in "interval" mode.
        run_immediately (bool): If True, runs an ingestion cycle right away on startup before entering sleep loop.
    """
    now_str = get_ethiopia_now().strftime("%Y-%m-%d %H:%M:%S %Z")

    print("=" * 60)
    print("🤖 Visit Ethiopia RAG Chatbot — Background Scheduler")
    print(f"🕒 Current Ethiopian Time: {now_str}")
    print(f"⏰ Mode: {mode.upper()} schedule")
    if mode in {"midnight", "target"}:
        time_label = f"{target_hour:02d}:{target_minute:02d}:00"
        print(f"🌙 Schedule Target: Daily at {time_label} (East Africa Time / UTC+3)")
    else:
        print(f"⏳ Schedule Interval: Every {interval_seconds / 3600.0:.2f} hour(s)")
    print(f"⚡ Run immediately on startup: {run_immediately}")
    print("=" * 60)

    urls = DEFAULT_URLS

    # Check whether an immediate initial crawl on launch was requested
    if run_immediately:
        print("⚡ Triggering initial crawl on startup...")
        execute_scheduled_crawl(urls)

    # Begin infinite loop for daemon scheduling
    while True:
        if mode in {"midnight", "target"}:
            sleep_duration, next_run_dt = get_seconds_until_next_target_time(
                target_hour=target_hour, target_minute=target_minute
            )
            next_run_str = next_run_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            if sleep_duration >= 3600:
                time_rem_str = f"~{sleep_duration / 3600.0:.2f} hours"
            else:
                time_rem_str = f"~{sleep_duration / 60.0:.1f} minutes ({sleep_duration:.0f} seconds)"
            print(f"⏳ [SCHEDULER] Sleeping until next target run ({next_run_str}) — in {time_rem_str}...")
        else:
            sleep_duration = interval_seconds
            next_run_dt = get_ethiopia_now() + datetime.timedelta(seconds=sleep_duration)
            next_run_str = next_run_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"⏳ [SCHEDULER] Sleeping until next scheduled run at: {next_run_str}...")

        # Chunk the sleep interval into short steps (1.0s) so daemon responds instantly to SIGINT / shutdown
        sleep_chunk = 1.0
        elapsed = 0.0
        try:
            while elapsed < sleep_duration:
                time.sleep(min(sleep_chunk, sleep_duration - elapsed))
                elapsed += sleep_chunk

        except KeyboardInterrupt:
            print("\n🛑 [SCHEDULER] Stopped by user (KeyboardInterrupt). Exiting cleanly.")
            sys.exit(0)

        # Trigger the next crawl cycle after waking up
        execute_scheduled_crawl(urls)


def start_scheduler_in_background(
    run_immediately: bool = False,
    mode: str = "midnight",
    target_hour: int = 0,
    target_minute: int = 0,
    interval_seconds: float = 86400.0,
) -> threading.Thread:
    """
    Launches the scheduler loop inside a non-blocking daemon thread.
    Intended for use inside api_server.py on server startup.
    """
    thread = threading.Thread(
        target=start_scheduler_loop,
        kwargs={
            "mode": mode,
            "target_hour": target_hour,
            "target_minute": target_minute,
            "interval_seconds": interval_seconds,
            "run_immediately": run_immediately,
        },
        daemon=True,
        name="BackgroundSchedulerThread",
    )
    thread.start()
    return thread


def main():
    """Parses CLI arguments and launches the background scheduler loop."""
    parser = argparse.ArgumentParser(
        description="Run automated web crawler background scheduler for Visit Ethiopia Chatbot."
    )

    parser.add_argument(
        "--mode",
        choices=["midnight", "target", "interval"],
        default=os.getenv("SCHEDULER_MODE", "midnight"),
        help="Scheduling mode: 'midnight' (00:00:00), 'target' (custom HH:MM), or 'interval' (fixed interval).",
    )

    parser.add_argument(
        "--target-time",
        type=str,
        default=None,
        help="Target daily time in 24h HH:MM format (e.g. 09:20, 15:30, 00:00). Overrides target-hour/minute.",
    )

    parser.add_argument(
        "--target-hour",
        type=int,
        default=int(os.getenv("SCHEDULER_TARGET_HOUR", "0")),
        help="Target daily hour (0-23, default: 0).",
    )

    parser.add_argument(
        "--target-minute",
        type=int,
        default=int(os.getenv("SCHEDULER_TARGET_MINUTE", "0")),
        help="Target daily minute (0-59, default: 0).",
    )

    parser.add_argument(
        "--run-initial",
        action="store_true",
        default=os.getenv("SCHEDULER_RUN_ON_STARTUP", "false").lower() in {"1", "true", "yes"},
        help="Run an immediate crawl cycle on startup before entering sleep loop (default: False).",
    )

    parser.add_argument(
        "--skip-initial",
        action="store_true",
        help="Skip the immediate initial crawl on launch and wait until the target time.",
    )

    parser.add_argument(
        "--interval-hours",
        type=float,
        default=float(os.getenv("CRAWL_INTERVAL_HOURS", "24")),
        help="Scheduling interval in hours (used when --mode interval is set, default: 24).",
    )

    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=None,
        help="Override interval in seconds (useful for quick testing/debugging).",
    )

    args = parser.parse_args()

    mode = args.mode
    target_hour = args.target_hour
    target_minute = args.target_minute

    if args.target_time:
        try:
            parts = args.target_time.split(":")
            target_hour = int(parts[0])
            target_minute = int(parts[1])
            if mode == "midnight":
                mode = "target"
        except Exception:
            print(f"⚠️ Invalid --target-time format '{args.target_time}'. Expected HH:MM (e.g. 09:20).")

    if args.interval_seconds is not None:
        interval = args.interval_seconds
    else:
        interval = args.interval_hours * 3600.0

    run_now = args.run_initial and not args.skip_initial

    start_scheduler_loop(
        mode=mode,
        target_hour=target_hour,
        target_minute=target_minute,
        interval_seconds=interval,
        run_immediately=run_now,
    )


if __name__ == "__main__":
    main()