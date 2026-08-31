import sys
from pathlib import Path

# src/data_prep.py imports `from utils.metrics import ...` assuming `src`
# is on sys.path (true when it's run directly as `python src/data_prep.py`).
# Mirror that for the test suite so it exercises the same import path.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
