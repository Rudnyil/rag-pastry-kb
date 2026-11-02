import os, sys
from pathlib import Path
os.environ.setdefault("EMBEDDER", "hashing")  # tests run offline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
