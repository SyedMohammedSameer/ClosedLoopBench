import sys
from pathlib import Path

# Make src/clbench importable from any test
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
