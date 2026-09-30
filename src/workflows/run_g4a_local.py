"""Windows/Conda entry point. Importing this file never schedules a solver."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from utils.g4a_local_execution import main
    raise SystemExit(main())
