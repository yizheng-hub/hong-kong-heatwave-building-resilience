"""Windows-safe entry point for the frozen six-event Open6 work order."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.e8_open6_execution import main


if __name__ == "__main__":
    raise SystemExit(main())
