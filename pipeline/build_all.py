"""
pipeline.build_all
------------------
Orchestrator for the whole pipeline. Phase 1 wires up the two stages that
exist so far; later phases append their stages here.

Usage:
    python -m pipeline.build_all            # incremental (reuse cached raw data)
    python -m pipeline.build_all --force    # refetch raw data and rebuild all
    python -m pipeline.build_all --only clean
"""

from __future__ import annotations

import os
# NLTK refuses proxied downloads by default; opt in (we declare the proxy trusted)
os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
try:
    import nltk
    nltk.pathsec.ALLOW_PROXIED_FETCH = True
except Exception:
    pass

import argparse
import sys
import time

os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
try:
    import nltk
    nltk.pathsec.ALLOW_PROXIED_FETCH = True
except Exception:
    pass

from . import aspect_tag, classify, clean, dimreduce, features, fetch_data, insights
from .common import CLEAN_CSV, RAW_CSV, ensure_dirs, logger

# Ordered stages: (name, callable). Appended as phases complete.
STAGES = [
    ("fetch_data", fetch_data.main),
    ("clean", clean.main),
    ("features", features.main),
    ("dimreduce", dimreduce.main),
    ("classify", classify.main),
    ("aspect_tag", aspect_tag.main),
    ("insights", insights.main),
]


def run(stage_names: list[str] | None = None, force: bool = False) -> int:
    ensure_dirs()
    selected = STAGES if not stage_names else [s for s in STAGES if s[0] in stage_names]
    if not selected:
        logger.error("No matching stages for %s", stage_names)
        return 1

    t0 = time.time()
    for name, func in selected:
        logger.info("=" * 70)
        logger.info("STAGE: %s", name)
        logger.info("=" * 70)
        # Pass an explicit argv so stage parsers never pick up build_all's flags.
        argv = ["--force"] if (force and name == "fetch_data") else []
        rc = func(argv)
        if rc not in (0, None):
            logger.error("Stage %s failed with code %s", name, rc)
            return rc

    logger.info("=" * 70)
    logger.info("Pipeline finished in %.1fs", time.time() - t0)
    logger.info("Artifacts: raw=%s cleaned=%s", RAW_CSV.exists(), CLEAN_CSV.exists())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full text-mining pipeline.")
    parser.add_argument("--force", action="store_true", help="Refetch raw data / rebuild all.")
    parser.add_argument("--only", nargs="+", metavar="STAGE", help="Run only these stages.")
    args = parser.parse_args(argv)
    return run(stage_names=args.only, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
