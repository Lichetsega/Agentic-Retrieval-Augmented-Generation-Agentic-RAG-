"""
Command-line ingestion runner.

Use this script to crawl configured tourism websites and push chunks
into the Chroma knowledge base before serving chatbot traffic.
"""

import argparse
import json
import os
import time
import traceback
from datetime import datetime

from utils import store_docs
from core_utils import DEFAULT_URLS

def run_ingestion(urls: list[str]) -> tuple[int, int]:
    """Run ingestion for each source URL and return (success_count, failed_count)."""
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    success = 0
    failed = 0
    details = []

    for url in urls:
        try:
            # store_docs() handles crawl + chunk + vector insert for one source.
            if store_docs(url):
                success += 1
                details.append({"url": url, "status": "success"})
            else:
                failed += 1
                details.append({"url": url, "status": "failed"})
        except Exception as e:
            print(f"❌ Ingestion failed for {url}: {e}")
            traceback.print_exc()
            failed += 1
            details.append({"url": url, "status": "error", "error": str(e)})

    duration_sec = round(time.time() - start_time, 2)

    # Save metadata log into data/crawl_history.json
    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        history_path = os.path.join(data_dir, "crawl_history.json")

        history_record = {
            "last_run_timestamp": start_timestamp,
            "duration_seconds": duration_sec,
            "total_urls": len(urls),
            "success_count": success,
            "failed_count": failed,
            "details": details
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_record, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to write crawl_history.json: {e}")

    return success, failed


def main():
    """CLI entrypoint used for one-off/manual indexing."""
    parser = argparse.ArgumentParser(description="Ingest tourism sources into Chroma.")
    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        help="Optional source URL(s). Pass multiple --url values.",
    )
    args = parser.parse_args()

    # If no custom URLs are passed, use the curated default source list.
    urls = args.urls if args.urls else DEFAULT_URLS
    success, failed = run_ingestion(urls)

    print(f"✅ Indexed sources: {success}")
    print(f"❌ Failed sources: {failed}")

    if success == 0 and failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
