"""Shared matplotlib style for Colosseum-V2 paper figures."""

import matplotlib
import matplotlib.font_manager as fm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT_FAMILY = "IBM Plex Mono"

if FONT_FAMILY not in {f.name for f in fm.fontManager.ttflist}:
    raise RuntimeError(
        f"Required font '{FONT_FAMILY}' is not installed. "
        "Install it system-wide (e.g. `sudo apt install fonts-ibm-plex`), then re-run."
    )

plt.rcParams["font.family"] = FONT_FAMILY
