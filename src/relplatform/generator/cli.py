"""python -m relplatform.generator.cli [--seed 42] [--months 12]"""
from __future__ import annotations

import argparse
import time

from relplatform.config import RANDOM_SEED, SIM_START_MONTHS_AGO
from relplatform.db import get_connection, reset_schema
from relplatform.generator.load import load
from relplatform.generator.simulate import simulate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--months", type=int, default=SIM_START_MONTHS_AGO)
    args = ap.parse_args()

    t0 = time.time()
    print(f"Simulating {args.months} months of data (seed={args.seed})...")
    result = simulate(seed=args.seed, months=args.months)
    print(
        f"  deployments={len(result.deployments)} incidents={len(result.incidents)} "
        f"alerts={len(result.alerts)} resource_metrics={len(result.resource_metrics)}"
    )

    con = get_connection()
    reset_schema(con)
    load(con, result)
    con.close()
    print(f"Loaded into DuckDB in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
