import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb
import pytest

from relplatform.db import init_schema
from relplatform.generator.simulate import simulate


@pytest.fixture(scope="session")
def sim_result():
    return simulate(seed=7, months=3)


@pytest.fixture()
def memdb():
    con = duckdb.connect(":memory:")
    init_schema(con)
    yield con
    con.close()
