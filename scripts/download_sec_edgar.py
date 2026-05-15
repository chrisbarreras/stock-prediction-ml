"""Download and extract the SEC EDGAR bulk company-facts dataset.

Fetches https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
(~1.5 GB compressed, expands to ~13,000 JSON files / ~15 GB on disk) and
extracts it to data/raw/kaggle/sec_edgar/companyfacts/, where
extract_financials.py expects to find it.

If the target directory already contains files, prompts whether to
re-download (overwriting) or skip. Use --force or --skip-if-exists for
non-interactive runs (e.g., from notebooks).

SEC requires a User-Agent identifying the requester. The default identifies
the project repo; set the SEC_USER_AGENT env var to override with your own
"Name email@domain.com" string.

Usage:
    python scripts/download_sec_edgar.py
    python scripts/download_sec_edgar.py --skip-if-exists
    python scripts/download_sec_edgar.py --force
"""

import argparse
import os
import sys
import zipfile

import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SEC_DIR = os.path.join(DATA_DIR, "raw", "kaggle", "sec_edgar")
FACTS_DIR = os.path.join(SEC_DIR, "companyfacts")
ZIP_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"

DEFAULT_UA = "stock-prediction-ml research " "(+https://github.com/Thomas-J-Barreras-Consulting/stock-prediction-ml)"
USER_AGENT = os.getenv("SEC_USER_AGENT", DEFAULT_UA)


def count_files(path):
    """Return number of regular files in path (0 if path missing)."""
    if not os.path.isdir(path):
        return 0
    return sum(1 for entry in os.scandir(path) if entry.is_file())


def prompt_overwrite(n_files):
    """Ask whether to re-download. Returns True to proceed, False to skip."""
    print(f"\nFound existing {FACTS_DIR}")
    print(f"  contains {n_files:,} files (~{n_files * 1.2 / 1024:.1f} GB if full dataset)")
    while True:
        resp = input("Re-download and overwrite? [y/N]: ").strip().lower()
        if resp in ("y", "yes"):
            return True
        if resp in ("", "n", "no"):
            return False
        print("Please answer y or n.")


def download_with_progress(url, dest_path):
    """Stream URL to dest_path, printing % progress on a single updating line."""
    print(f"\nDownloading {url}")
    print(f"  -> {dest_path}")
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    tmp_path = dest_path + ".tmp"
    try:
        with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            written = 0
            last_pct = -1
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = int(100 * written / total)
                        if pct != last_pct:
                            mb_w = written / (1024 * 1024)
                            mb_t = total / (1024 * 1024)
                            print(
                                f"\r  [{pct:3d}%] {mb_w:7,.1f} / {mb_t:7,.1f} MB",
                                end="",
                                flush=True,
                            )
                            last_pct = pct
            print()
        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print(f"  Saved {os.path.getsize(dest_path) / (1024 * 1024):,.1f} MB")


def extract_with_progress(zip_path, dest_dir):
    """Extract every member of zip_path into dest_dir with running file count."""
    print(f"\nExtracting {zip_path}")
    print(f"  -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        total = len(members)
        for i, name in enumerate(members, 1):
            zf.extract(name, dest_dir)
            if i % 500 == 0 or i == total:
                pct = 100 * i / total
                print(
                    f"\r  [{pct:5.1f}%] {i:,} / {total:,} files",
                    end="",
                    flush=True,
                )
        print()
    print(f"  Extracted {total:,} files")


def main():
    parser = argparse.ArgumentParser(
        description="Download SEC EDGAR companyfacts.zip and extract it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Always download and overwrite, even if companyfacts/ already populated",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip silently if companyfacts/ already has files (for non-interactive use)",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip after extraction (default: delete to save ~1.5 GB)",
    )
    args = parser.parse_args()

    if args.force and args.skip_if_exists:
        print("ERROR: --force and --skip-if-exists are mutually exclusive.", file=sys.stderr)
        return 2

    print("=== SEC EDGAR companyfacts downloader ===")
    print(f"Target dir: {FACTS_DIR}")
    print(f"User-Agent: {USER_AGENT}")

    existing = count_files(FACTS_DIR)
    if existing > 0:
        if args.skip_if_exists:
            print(f"\ncompanyfacts/ already has {existing:,} files; --skip-if-exists set, skipping.")
            return 0
        if args.force:
            print(f"\ncompanyfacts/ has {existing:,} files; --force set, re-downloading.")
        else:
            if not prompt_overwrite(existing):
                print("Skipping download (existing data preserved).")
                return 0

    os.makedirs(SEC_DIR, exist_ok=True)
    zip_path = os.path.join(SEC_DIR, "companyfacts.zip")

    download_with_progress(ZIP_URL, zip_path)
    extract_with_progress(zip_path, FACTS_DIR)

    if not args.keep_zip:
        os.remove(zip_path)
        print(f"  Removed {zip_path} (use --keep-zip to retain)")

    final_count = count_files(FACTS_DIR)
    print(f"\nDone! {final_count:,} company facts files in {FACTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
