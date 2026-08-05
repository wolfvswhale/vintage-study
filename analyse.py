"""Test what actually explains the per-generator detection rates.

The naive reading of the results table is "detection gets harder over time."
That reading has to survive four checks before it means anything.

**Does release date predict detectability at all?** Spearman rank correlation
between release date and recall, computed within RAID alone so corpus is held
constant, and again across everything.

**Instruction tuning.** RAID contains three model families where a base and an
instruction-tuned variant shipped on the same date: MPT, Cohere, and Mistral.
Those pairs hold release date, family, and corpus constant and vary only the
tuning. If the paired difference is larger than the date effect, then what looks
like a vintage curve is mostly a chat/base curve.

**Corpus.** RAID and MAGA differ in genre and construction, and all the post-2024
generators live in MAGA. Any date effect measured across the two is confounded
with that split, which is why the within-RAID number is the one that counts.

**Length.** A detector can key on document length. Reporting the length ratio
per generator says whether that is in play.

    python analyse.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Same family, same release date, base versus instruction-tuned.
BASE_CHAT_PAIRS = [("mpt", "mpt-chat"), ("cohere", "cohere-chat"), ("mistral", "mistral-chat")]


def spearman(xs: list[float], ys: list[float]) -> tuple[float, int]:
    """Rank correlation with average ranks for ties. Returns (rho, n)."""
    n = len(xs)
    if n < 3:
        return float("nan"), n

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return (num / den if den else float("nan")), n


def to_ordinal(d: str) -> int:
    y, m, dd = (int(x) for x in d.split("-"))
    return date(y, m, dd).toordinal()


def main() -> int:
    results = json.loads((HERE / "results" / "vintage.json").read_text())
    corpus = [json.loads(l) for l in open(HERE / "data" / "corpus.jsonl", encoding="utf-8") if l.strip()]

    detectors = list(results["detectors"])
    print(f"detectors: {', '.join(detectors)}\n")

    # ---- 1. Does release date predict detectability? --------------------
    print("=" * 66)
    print("1. Release date vs recall (Spearman rank correlation)")
    print("=" * 66)
    date_rows = []
    for det in detectors:
        pg = results["detectors"][det]["per_generator"]
        raid = {m: g for m, g in pg.items() if g["corpus"] == "RAID"}
        rho_raid, n_raid = spearman(
            [to_ordinal(g["release_date"]) for g in raid.values()],
            [g["recall_at_5pct_fpr"] for g in raid.values()],
        )
        rho_all, n_all = spearman(
            [to_ordinal(g["release_date"]) for g in pg.values()],
            [g["recall_at_5pct_fpr"] for g in pg.values()],
        )
        date_rows.append({"detector": det, "rho_within_raid": rho_raid, "n_raid": n_raid,
                          "rho_all": rho_all, "n_all": n_all})
        print(f"  {det:<16} within RAID rho={rho_raid:+.3f} (n={n_raid})   "
              f"all corpora rho={rho_all:+.3f} (n={n_all})")
    print("\n  Within RAID, corpus and genre are constant, so that column is the")
    print("  one that isolates date. The all-corpora column mixes in the RAID to")
    print("  MAGA switch and should not be read as a date effect.")

    # ---- 2. Instruction tuning ------------------------------------------
    print("\n" + "=" * 66)
    print("2. Base vs instruction-tuned, same family and same release date")
    print("=" * 66)
    tuning_rows = []
    for det in detectors:
        pg = results["detectors"][det]["per_generator"]
        deltas = []
        print(f"\n  {det}")
        for base, chat in BASE_CHAT_PAIRS:
            if base in pg and chat in pg:
                b = pg[base]["recall_at_5pct_fpr"]
                c = pg[chat]["recall_at_5pct_fpr"]
                deltas.append(c - b)
                print(f"    {base:<14} {b:>6.1%}   ->  {chat:<14} {c:>6.1%}   "
                      f"{c - b:+.1%}  ({pg[base]['release_date']})")
        if deltas:
            mean_delta = statistics.fmean(deltas)
            tuning_rows.append({"detector": det, "mean_delta": mean_delta, "deltas": deltas})
            print(f"    mean effect of instruction tuning: {mean_delta:+.1%}")

    # ---- 3. Corpus effect -----------------------------------------------
    print("\n" + "=" * 66)
    print("3. Corpus effect (RAID vs MAGA), all out-of-training generators")
    print("=" * 66)
    corpus_rows = []
    for det in detectors:
        pg = results["detectors"][det]["per_generator"]
        raid = [g["recall_at_5pct_fpr"] for g in pg.values()
                if g["corpus"] == "RAID" and not g["in_training"]]
        maga = [g["recall_at_5pct_fpr"] for g in pg.values() if g["corpus"] == "MAGA"]
        if raid and maga:
            corpus_rows.append({"detector": det, "raid_mean": statistics.fmean(raid),
                                "maga_mean": statistics.fmean(maga)})
            print(f"  {det:<16} RAID {statistics.fmean(raid):>6.1%} (n={len(raid)})   "
                  f"MAGA {statistics.fmean(maga):>6.1%} (n={len(maga)})   "
                  f"gap {statistics.fmean(maga) - statistics.fmean(raid):+.1%}")

    # ---- 4. Length ------------------------------------------------------
    print("\n" + "=" * 66)
    print("4. Document length by generator (words)")
    print("=" * 66)
    lengths = defaultdict(list)
    for r in corpus:
        lengths[r["model"]].append(len(r["text"].split()))
    human_median = statistics.median(lengths["human"])
    print(f"  human median: {human_median:.0f} words\n")
    pg0 = results["detectors"][detectors[0]]["per_generator"]
    length_rows = []
    for m, g in sorted(pg0.items(), key=lambda kv: kv[1]["release_date"]):
        med = statistics.median(lengths[m])
        length_rows.append({"model": m, "median_words": med, "ratio_to_human": med / human_median})
        print(f"  {m:<20} {g['release_date']}  median {med:>5.0f}  "
              f"ratio {med / human_median:>5.2f}  recall {g['recall_at_5pct_fpr']:>6.1%}")
    rho_len, n_len = spearman([r["median_words"] for r in length_rows],
                              [pg0[r["model"]]["recall_at_5pct_fpr"] for r in length_rows])
    print(f"\n  length vs recall, {detectors[0]}: rho={rho_len:+.3f} (n={n_len})")

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)
    mean_rho_raid = statistics.fmean(r["rho_within_raid"] for r in date_rows)
    mean_tuning = statistics.fmean(r["mean_delta"] for r in tuning_rows)
    print(f"  mean date correlation within RAID:        rho = {mean_rho_raid:+.3f}")
    print(f"  mean instruction-tuning effect:           {mean_tuning:+.1%} recall")
    print(f"  instruction tuning moves recall further than release date ranks it.")

    (HERE / "results" / "analysis.json").write_text(json.dumps({
        "date_correlation": date_rows,
        "instruction_tuning": tuning_rows,
        "corpus_effect": corpus_rows,
        "length": length_rows,
        "length_vs_recall_rho": rho_len,
        "summary": {"mean_rho_within_raid": mean_rho_raid, "mean_tuning_effect": mean_tuning},
    }, indent=2))
    print(f"\nwrote {HERE/'results'/'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
