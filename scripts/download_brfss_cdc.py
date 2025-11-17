# Author: Prajan V (Infosys Springboard 6.0)
# Date: 2025-11-17

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Iterable
import zipfile

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brfss_cdc_downloader")

CDC_BASE = "https://www.cdc.gov/brfss/annual_data/{year}/files/"
CANDIDATE_FILENAMES = [
    "LLCP{year}XPT.zip",
    "LLCP{year}_XPT.ZIP",
]


def ensure_dirs(root: Path) -> tuple[Path, Path]:
    data_brfss = root / "data" / "brfss"
    raw = data_brfss / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    return data_brfss, raw


def download_year(year: int, raw_dir: Path, timeout: int = 60) -> Path | None:
    for fname in CANDIDATE_FILENAMES:
        url = CDC_BASE.format(year=year) + fname.format(year=year)
        try:
            logger.info("Downloading %s", url)
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                logger.warning("HTTP %s at %s", r.status_code, url)
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                names = zf.namelist()
                xpt_name = None
                for n in names:
                    if n.lower().endswith(".xpt"):
                        xpt_name = n
                        break
                if xpt_name is None:
                    logger.warning("No XPT inside zip for %s", year)
                    continue
                target = raw_dir / Path(xpt_name).name
                logger.info("Extracting %s -> %s", xpt_name, target)
                with zf.open(xpt_name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                return target
        except requests.RequestException as e:
            logger.warning("Failed to download %s: %s", url, e)
            continue
        except zipfile.BadZipFile:
            logger.warning("Bad ZIP at %s", url)
            continue
    logger.error("Failed to acquire BRFSS %s from CDC", year)
    return None


def main(years: Iterable[int] = (2011, 2012, 2013, 2014, 2015)):
    root = Path(__file__).resolve().parents[1]
    _, raw = ensure_dirs(root)
    results = []
    for y in years:
        p = download_year(y, raw)
        results.append((y, p))
    ok = [y for y, p in results if p is not None]
    logger.info("Downloaded XPT for years: %s", ok)


if __name__ == "__main__":
    main()
