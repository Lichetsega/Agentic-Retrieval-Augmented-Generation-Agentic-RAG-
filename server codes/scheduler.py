"""
24-Hour Automated Web Crawling Scheduler for Visit Ethiopia RAG Chatbot.

This daemon script automatically executes web crawling and vector database
re-indexing every 24 hours (or configured interval), keeping ChromaDB up to date.

Usage:
    python scheduler.py                       # Starts crawling immediately, then every 24h
    python scheduler.py --skip-initial         # Sleeps first, then crawls every 24h
    python scheduler.py --interval-seconds 10  # Test mode (runs every 10 sec)
"""

import argparse
import datetime
import os
import sys
import time
import traceback
from dotenv import load_dotenv

# Ensure local module imports work seamlessly regardless of the directory from which the script is launched
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from core_utils import DEFAULT_URLS
from ingest import run_ingestion


def execute_scheduled_crawl(urls: list[str]) -> bool:
    """
    Executes a single web crawl and vector indexing cycle.

    Args:
        urls (list[str]): List of target website URLs to crawl and process.

    Returns:
        bool: True if the ingestion job ran without unhandled exceptions, False otherwise.
    """
    # Format the current local date and time as a human-readable timestamp string
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Print execution start banner and metadata
    print("\n" + "=" * 60)
    print(f"🚀 [SCHEDULER] Starting automated 24h crawl cycle at {now_str}")
    print(f"🌐 Target sources: {len(urls)} URLs")
    print("=" * 60)

    try:
        # Trigger the ingestion pipeline (crawling, text extraction, chunking, embedding, vector DB update)
        success, failed = run_ingestion(urls)
        finish_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Print execution summary banner and statistics
        print("\n" + "-" * 60)
        print(f"✅ [SCHEDULER] Crawl completed at {finish_str}")
        print(f"📊 Results: {success} succeeded, {failed} failed")
        print("-" * 60 + "\n")
        return True

    except Exception as e:
        # Catch, log, and print traceback for any unexpected runtime errors to prevent the daemon from crashing
        print(f"❌ [SCHEDULER ERROR] Unhandled exception during crawl: {e}")
        traceback.print_exc()
        return False


def start_scheduler_loop(interval_seconds: float, run_immediately: bool = False):
    """
    Main infinite scheduler loop that manages timing, initial executions, and sleep cycles.

    Args:
        interval_seconds (float): Duration in seconds between recurring crawl cycles.
        run_immediately (bool): If True, runs an ingestion cycle right away before entering the sleep loop.
    """
    # Convert seconds to hours for cleaner display logging
    interval_hours = interval_seconds / 3600.0

    # Print startup configuration details
    print("=" * 60)
    print("🤖 Visit Ethiopia RAG Chatbot — Background Scheduler")
    print(f"⏰ Interval: Every {interval_hours:.2f} hour(s) ({interval_seconds:.0f} seconds)")
    print(f"⚡ Run immediately on startup: {run_immediately}")
    print("=" * 60)

    # Use default target URLs imported from ingest.py
    urls = DEFAULT_URLS

    # Check whether an immediate initial crawl on launch was requested
    if run_immediately:
        print("⚡ Triggering initial crawl on startup...")
        execute_scheduled_crawl(urls)

    # Begin infinite loop for daemon scheduling
    while True:
        # Calculate expected timestamp for the next scheduled execution cycle
        next_run_dt = datetime.datetime.now() + datetime.timedelta(seconds=interval_seconds)
        next_run_str = next_run_dt.strftime("%Y-%m-%d %H:%M:%S")
        print(f"⏳ [SCHEDULER] Sleeping until next scheduled run at: {next_run_str}...")

        # Chunk the sleep interval into short steps (5.0s) so the daemon responds instantly to SIGINT / Ctrl+C
        sleep_chunk = 5.0
        elapsed = 0.0
        try:
            # Incrementally sleep until the full target interval has elapsed
            while elapsed < interval_seconds:
                # Sleep for either the 5s chunk or remaining balance, whichever is smaller
                time.sleep(min(sleep_chunk, interval_seconds - elapsed))
                elapsed += sleep_chunk

        except KeyboardInterrupt:
            # Intercept manual user interruption cleanly and exit gracefully
            print("\n🛑 [SCHEDULER] Stopped by user (KeyboardInterrupt). Exiting cleanly.")
            sys.exit(0)

        # Trigger the next crawl cycle after waking up
        execute_scheduled_crawl(urls)


def main():
    """
    Parses CLI arguments and launches the background scheduler loop with configured parameters.
    """
    # Initialize argument parser with utility description
    parser = argparse.ArgumentParser(
        description="Run automated 24-hour web crawler background scheduler."
    )

    # Flag argument: Optional bypass for the initial execution on launch
    parser.add_argument(
        "--skip-initial",
        action="store_true",
        help="Skip the immediate initial crawl on launch and wait for the first 24h interval.",
    )

    # Option argument: Set custom crawl interval in hours (defaults to .env value or 24)
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=float(os.getenv("CRAWL_INTERVAL_HOURS", "24")),
        help="Scheduling interval in hours (default: 24).",
    )

    # Option argument: Set custom crawl interval in seconds for quick testing/debugging
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=None,
        help="Override interval in seconds (useful for quick testing/debugging).",
    )

    # Parse all passed command-line arguments
    args = parser.parse_args()

    # Determine final interval in seconds: explicit seconds override takes precedence over hours
    if args.interval_seconds is not None:
        interval = args.interval_seconds
    else:
        interval = args.interval_hours * 3600.0

    # Execute immediately unless --skip-initial is explicitly set by the user
    run_now = not args.skip_initial

    # Start the continuous scheduler loop
    start_scheduler_loop(interval_seconds=interval, run_immediately=run_now)


# Entry point guard: run main() only when script is executed directly from CLI
if __name__ == "__main__":
    main()