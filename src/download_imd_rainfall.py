"""Download and validate the official IMD 0.25-degree daily rainfall files."""
from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ARCHIVE_URL = "https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html"
ENDPOINT = "https://imdpune.gov.in/cmpg/Griddata/RF25.php"
RAW_DIR = Path("data/raw/imd_rainfall")


def advertised_years(session: requests.Session) -> list[int]:
    response = session.get(ARCHIVE_URL, timeout=60)
    response.raise_for_status()
    years = sorted({int(y) for y in re.findall(r"(?:RF25\s*=\s*|ind)(19\d{2}|20\d{2})", response.text, re.I)})
    # The archive page may implement years in JavaScript or a POST form; this range is
    # constrained by the archive's stated 1901–2024 coverage and checked file-by-file.
    return years or list(range(1901, datetime.now(timezone.utc).year))


def is_netcdf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            magic = stream.read(8)
        return magic.startswith(b"CDF") or magic == b"\x89HDF\r\n\x1a\n"
    except OSError:
        return False


def download_years(years: list[int], out_dir: Path = RAW_DIR, retries: int = 3) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "rice-monsoon-research/1.0 (reproducible research)"})
    log_path = out_dir / "download_log.csv"
    cached_by_year = {year: next((p for p in out_dir.glob("*.nc") if str(year) in p.name and is_netcdf(p)), None) for year in years}
    if all(cached_by_year.values()):
        available = set()
        if log_path.exists():
            try:
                with log_path.open(newline="", encoding="utf-8") as stream:
                    for record in csv.DictReader(stream):
                        available.update(int(y) for y in record.get("archive_page_years_seen", "").split(";") if y.isdigit())
            except (OSError, csv.Error):
                pass
    else:
        available = set(advertised_years(session))
    fields = ["year", "timestamp_utc", "request_url", "status", "file", "bytes", "message", "archive_page_years_seen"]
    logs = []
    paths = []
    for year in years:
        # Some IMD page versions expose only a partial rendered list. Treat it as
        # discovery metadata, not an allowlist; validate each requested response.
        existing = cached_by_year[year]
        if existing:
            paths.append(existing)
            logs.append([year, datetime.now(timezone.utc).isoformat(), ENDPOINT + f" [POST RF25={year}; cached locally]", "cached_valid", existing.name, existing.stat().st_size, "", ";".join(map(str, sorted(available)))])
            continue
        last_error = "unknown error"
        for attempt in range(retries):
            try:
                response = session.post(ENDPOINT, data={"RF25": str(year)}, timeout=(30, 240), stream=True)
                response.raise_for_status()
                disposition = response.headers.get("Content-Disposition", "")
                match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)', disposition, re.I)
                name = Path(match.group(1)).name if match else f"RF25_ind{year}_rfp25.nc"
                if not name.lower().endswith((".nc", ".nc4")):
                    name = f"RF25_ind{year}_rfp25.nc"
                target = out_dir / name
                temp = target.with_suffix(target.suffix + ".part")
                byte_count = 0
                with temp.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            byte_count += len(chunk)
                if byte_count == 0 or not is_netcdf(temp):
                    temp.unlink(missing_ok=True)
                    raise RuntimeError("response is empty or not a NetCDF file")
                temp.replace(target)
                paths.append(target)
                logs.append([year, datetime.now(timezone.utc).isoformat(), ENDPOINT + f" [POST RF25={year}]", "downloaded", target.name, target.stat().st_size, "", ";".join(map(str, sorted(available)))])
                break
            except Exception as exc:  # retry transient HTTP and transport errors
                last_error = str(exc)
                if attempt + 1 < retries:
                    time.sleep(2 ** attempt)
        else:
            logs.append([year, datetime.now(timezone.utc).isoformat(), ENDPOINT + f" [POST RF25={year}]", "failed", "", 0, last_error, ";".join(map(str, sorted(available)))])
            with log_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                if stream.tell() == 0:
                    writer.writerow(fields)
                writer.writerows(logs)
            raise RuntimeError(f"Could not obtain valid IMD NetCDF for {year}: {last_error}")
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(logs)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("years", nargs="+", type=int)
    args = parser.parse_args()
    for path in download_years(args.years):
        print(path)


if __name__ == "__main__":
    main()
