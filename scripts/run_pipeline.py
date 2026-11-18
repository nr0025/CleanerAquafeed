"""Run the screening framework.

Example:
    python scripts/run_pipeline.py --config config/default.toml --out results
    python scripts/run_pipeline.py --stages formulation uncertainty benchmarks growth figures tables
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aquafeed_screen.pipeline import STAGES, Run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/default.toml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    a = ap.parse_args()
    Run(a.config, a.out).run(a.stages, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
