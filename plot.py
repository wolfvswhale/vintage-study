"""The figure.

Two panels, because the result is a negative and a positive and they need
different forms.

Left: recall against release date. The job is to show there is no downward
trend, so it is a scatter with the points labelled, not a line. Corpus is
encoded by colour and by marker shape, so identity never depends on colour
alone. The RAID-to-MAGA boundary is drawn because that switch, not the passage
of time, is what the pooled correlation is really measuring.

Right: the three base-to-chat pairs as slopes. The job is change on a shared
scale, which is what a slope chart is for. All three go the same direction.

Palette slots are the validated categorical blue and orange (checked with the
dataviz validator: CVD ΔE 24.7 protan, normal-vision ΔE 33.6, both PASS).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
RAID_C = "#2a78d6"
MAGA_C = "#eb6834"

PAIRS = [("mpt", "mpt-chat"), ("cohere", "cohere-chat"), ("mistral", "mistral-chat")]
DETECTOR = "char tf-idf"

# Hand-placed label offsets. The 2023 generators pile into a narrow date band at
# similar recall, so automatic placement collides; these are set by looking at
# the rendered figure rather than by trusting the default.
LABEL_OFFSET = {
    "gpt2": (0, 12), "gpt3": (0, 12), "cohere": (0, -18), "mpt": (-8, -18),
    "mpt-chat": (-46, 8), "chatgpt": (-16, 16), "gpt4": (4, 15),
    "cohere-chat": (-10, -4), "llama-chat": (38, -2), "mistral": (2, -18),
    "mistral-chat": (40, -6),
    "gpt-4o-mini": (-4, -18), "llama-3.1-8b": (-2, -18), "deepseek-v3": (-16, 12),
    "gemini-2.0-flash": (-8, -18), "qwen3-8b": (16, 10), "qwen3-plus": (14, -16),
    "mistral-medium": (16, 8),
}
LABEL_HA = {
    "mpt-chat": "right", "llama-chat": "left", "cohere-chat": "right",
    "mistral-chat": "left", "qwen3-8b": "left", "qwen3-plus": "left",
    "mistral-medium": "left", "cohere-chat": "left",
}


def ordinal(d: str) -> int:
    y, m, dd = (int(x) for x in d.split("-"))
    return date(y, m, dd).toordinal()


def main() -> int:
    res = json.loads((HERE / "results" / "vintage.json").read_text())
    ana = json.loads((HERE / "results" / "analysis.json").read_text())
    pg = res["detectors"][DETECTOR]["per_generator"]
    rho = next(r["rho_within_raid"] for r in ana["date_correlation"] if r["detector"] == DETECTOR)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 6.0), gridspec_kw={"width_ratios": [1.75, 1]}
    )
    fig.patch.set_facecolor(SURFACE)

    # ---------------- Panel 1: recall vs release date ----------------
    ax1.set_facecolor(SURFACE)
    for name, g in pg.items():
        x = ordinal(g["release_date"])
        y = g["recall_at_5pct_fpr"] * 100
        is_raid = g["corpus"] == "RAID"
        colour = RAID_C if is_raid else MAGA_C
        marker = "o" if is_raid else "^"
        ax1.scatter(x, y, s=110, color=colour, marker=marker, zorder=3,
                    edgecolor=SURFACE, linewidth=2)
        dx, dy = LABEL_OFFSET.get(name, (0, 11))
        ax1.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                     ha=LABEL_HA.get(name, "center"), fontsize=8.5, color=INK2)

    boundary = ordinal("2024-03-01")
    ax1.axvline(boundary, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax1.text(boundary, 96, "  corpus changes here\n  (RAID → MAGA)", fontsize=8.5,
             color=MUTED, va="top", ha="left")

    ticks = [ordinal(f"{y}-01-01") for y in range(2019, 2026)]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([str(y) for y in range(2019, 2026)], fontsize=9.5, color=INK2)
    ax1.set_ylim(0, 100)
    ax1.set_yticks(range(0, 101, 20))
    ax1.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], fontsize=9.5, color=INK2)
    ax1.set_xlabel("generator release date", fontsize=10.5, color=INK2, labelpad=9)
    ax1.set_ylabel("detection rate at a fixed 5% false-positive rate",
                   fontsize=10.5, color=INK2, labelpad=9)
    ax1.set_title(
        f"Release date does not predict detectability\n"
        f"within RAID, where corpus is held constant: Spearman rho = {rho:+.2f}",
        fontsize=12.5, color=INK, loc="left", pad=14, linespacing=1.5,
    )
    ax1.grid(axis="y", color="#e8e8e4", linewidth=1, zorder=0)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax1.spines[side].set_color("#dcdcd7")
    ax1.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=RAID_C, markersize=9,
                   label="RAID corpus"),
            Line2D([], [], marker="^", linestyle="", color=MAGA_C, markersize=9,
                   label="MAGA-Bench corpus"),
        ],
        frameon=False, fontsize=9.5, labelcolor=INK2, loc="lower left",
    )

    # ---------------- Panel 2: instruction tuning ----------------
    ax2.set_facecolor(SURFACE)
    for i, (base, chat) in enumerate(PAIRS):
        yb = pg[base]["recall_at_5pct_fpr"] * 100
        yc = pg[chat]["recall_at_5pct_fpr"] * 100
        ax2.plot([0, 1], [yb, yc], color=RAID_C, linewidth=2, zorder=2, alpha=0.85)
        ax2.scatter([0, 1], [yb, yc], s=95, color=RAID_C, zorder=3,
                    edgecolor=SURFACE, linewidth=2)
        # cohere and mistral sit within a point of each other at the base end.
        nudge = {"cohere": 9, "mistral": -9}.get(base, 0)
        ax2.annotate(f"{base}  {yb:.0f}%", (0, yb), textcoords="offset points",
                     xytext=(-11, nudge), ha="right", va="center", fontsize=9, color=INK2)
        ax2.annotate(f"{yc:.0f}%", (1, yc), textcoords="offset points",
                     xytext=(11, 0), ha="left", va="center", fontsize=9, color=INK2)

    ax2.set_xlim(-0.62, 1.42)
    ax2.set_ylim(0, 100)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["base", "instruction-tuned"], fontsize=10, color=INK2)
    ax2.set_yticks(range(0, 101, 20))
    ax2.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], fontsize=9.5, color=INK2)
    mean_delta = next(r["mean_delta"] for r in ana["instruction_tuning"]
                      if r["detector"] == DETECTOR)
    ax2.set_title(
        f"Instruction tuning does\n"
        f"same family, same release date: {mean_delta * 100:+.0f} points",
        fontsize=12.5, color=INK, loc="left", pad=14, linespacing=1.5,
    )
    ax2.grid(axis="y", color="#e8e8e4", linewidth=1, zorder=0)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax2.spines[side].set_color("#dcdcd7")

    fig.text(0.012, 0.018,
             f"Detector: {DETECTOR}, trained only on generators released before "
             f"{res['train_before']}. Threshold fixed so 5% of held-out human documents "
             f"are flagged; recall measured at that threshold.\n"
             f"Sources: RAID (Dugan et al., ACL 2024) and MAGA-Bench, "
             f"18 generators, 15,500 documents. Code and data: "
             f"github.com/wolfvswhale/vintage-study",
             fontsize=8, color=MUTED, linespacing=1.6)

    fig.tight_layout(rect=(0, 0.075, 1, 1))
    out_png = HERE / "results" / "vintage.png"
    out_svg = HERE / "results" / "vintage.svg"
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    fig.savefig(out_svg, facecolor=SURFACE)
    print(f"wrote {out_png}")
    print(f"wrote {out_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
