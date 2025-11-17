# Author: Prajan V (Infosys Springboard 6.0)
# Date: 2025-11-17

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import kagglehub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brfss_khub")


def main():
    root = Path(__file__).resolve().parents[1]
    data_brfss = root / "data" / "brfss"
    data_brfss.mkdir(parents=True, exist_ok=True)

    path = kagglehub.dataset_download("cdc/behavioral-risk-factor-surveillance-system")
    base = Path(path)
    logger.info("Downloaded Kaggle dataset to %s", base)

    targets = {2011: None, 2012: None, 2013: None, 2014: None, 2015: None}

    # Prefer CSV; fall back to XPT
    candidates = list(base.rglob("*"))
    for y in list(targets.keys()):
        # CSV first
        for p in candidates:
            name = p.name.lower()
            if name.endswith(".csv") and str(y) in name and p.stat().st_size > 1024:
                targets[y] = p
                break
        if targets[y] is not None:
            continue
        # XPT fallback
        for p in candidates:
            name = p.name.lower()
            if name.endswith(".xpt") and str(y) in name and p.stat().st_size > 1024:
                targets[y] = p
                break

    for y, p in targets.items():
        if p is None:
            logger.warning("No file found for %s", y)
            continue
        dest = data_brfss / (f"{y}.csv" if p.suffix.lower() == ".csv" else ("raw/" + p.name))
        dest_path = data_brfss / dest
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Copying %s -> %s", p, dest_path)
        shutil.copyfile(p, dest_path)


if __name__ == "__main__":
    main()
