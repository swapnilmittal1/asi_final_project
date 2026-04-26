"""
One-shot builder for the final severity / harm package.

This keeps the final project workflow reproducible by chaining the exact
artifacts used in the report narrative.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import List


logger = logging.getLogger(__name__)

STEPS: List[List[str]] = [
    [sys.executable, "-m", "src.data.build_severity_tables"],
    [sys.executable, "-m", "src.data.build_harm_scenarios"],
    [sys.executable, "-m", "src.experiments.run_harm_method_comparison"],
    [sys.executable, "-m", "src.experiments.run_severity_sensitivity"],
    [sys.executable, "-m", "src.experiments.run_harm_story_sweep"],
    [sys.executable, "-m", "src.experiments.make_harm_figures"],
    [sys.executable, "-m", "src.experiments.make_harm_story_figures"],
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for step in STEPS:
        logger.info("Running: %s", " ".join(step))
        subprocess.run(step, check=True)
    logger.info("Final harm package complete.")


if __name__ == "__main__":
    main()
