"""
Command-line ingestion runner.

Use this script to crawl configured tourism websites and push chunks
into the Chroma knowledge base before serving chatbot traffic.
"""

import argparse
import traceback

from utils import store_docs

DEFAULT_URLS = [
    "https://visitethiopia.et/",
    "https://visitoromia.org/",
    "https://visitamhara.travel/",
    "https://tourismtigrai.com/",
    "https://visitsidama.travel/",
    "https://visitsouthethiopia.et/",
    "https://www.pmo.gov.et/",
    "https://mot.gov.et/",
    "https://mfa.gov.et/",
    "https://www.mor.gov.et/",
    "https://motri.gov.et/en",
    "https://www.motl.gov.et/en",
    "https://www.mofed.gov.et/",
    "https://www.moi.gov.et/",
    "http://www.mint.gov.et/",
    "https://mopd.gov.et/en/",
    "https://mui.gov.et/",
    "https://www.mowe.gov.et/en/",
    "https://www.mowsa.gov.et/",
    "https://www.moh.gov.et/",
    "https://www.ethiopianairlines.com/",
    "https://www.ethiopianholidays.com/",
    "https://ics.gov.et/",
    "https://ecc.gov.et/",
    "https://www.moh.gov.et/",
    "https://combanketh.et/",
    "https://nbe.gov.et/",
    "https://www.etoa.travel/",
    "https://www.stoa-ethiopia.org/",
    "https://www.aha97.com/",
    "https://www.ethiopianrun.org/",
]

def run_ingestion(urls: list[str]) -> tuple[int, int]:
    """Run ingestion for each source URL and return (success_count, failed_count)."""
    success = 0
    failed = 0
    for url in urls:
        try:
            # store_docs() handles crawl + chunk + vector insert for one source.
            if store_docs(url):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Ingestion failed for {url}: {e}")
            traceback.print_exc()
            failed += 1
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
