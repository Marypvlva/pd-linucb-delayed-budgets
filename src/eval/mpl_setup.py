from __future__ import annotations

import os
from pathlib import Path
from tempfile import gettempdir


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str((Path(gettempdir()) / "pd_linucb_mplconfig").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
