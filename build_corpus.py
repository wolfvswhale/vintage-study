"""Assemble a generator-vintage-stratified corpus.

The question this exists to answer: does a detector's accuracy fall as the
generator that produced the text gets newer? Everyone assumes it does. The
literature always frames it as cross-generator generalisation or distribution
shift, never as a function of release date, so nobody has drawn the curve.

The data to draw it already exists. RAID labels every generation with the model
that produced it, spanning GPT-2 through gpt-4-0613. MAGA-Bench extends the
range through 2026. Attaching a public release date to each generator and
cutting the results that way is the whole experiment.

Sampling is reservoir-based per (model, domain, decoding) cell so the result
does not depend on how the source file happens to be ordered.

    python build_corpus.py --per-cell 60
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"

# Public first-release dates. Where a model was released in stages, the date is
# the one for the specific checkpoint RAID or MAGA names. Sources recorded in
# RELEASE_NOTES so the mapping can be argued with rather than trusted.
RELEASE = {
    # RAID generators
    "gpt2": "2019-02-14",
    "gpt3": "2022-01-27",          # text-davinci-002 lineage (InstructGPT)
    "chatgpt": "2023-06-13",       # gpt-3.5-turbo-0613
    "gpt4": "2023-06-13",          # gpt-4-0613
    "cohere": "2022-11-15",
    "cohere-chat": "2023-07-13",
    "mpt": "2023-05-05",
    "mpt-chat": "2023-05-05",
    "llama-chat": "2023-07-18",    # Llama-2-70b-chat
    "mistral": "2023-09-27",       # Mistral-7B
    "mistral-chat": "2023-09-27",
    # MAGA generators
    "gpt-4o-mini": "2024-07-18",
    "llama-3.1-8b": "2024-07-23",
    "gemini-2.0-flash": "2025-02-05",
    "deepseek-v3": "2024-12-26",
    "qwen3-8b": "2025-04-29",
    "qwen3-plus": "2025-04-29",
    "mistral-medium": "2025-05-07",
    "deepseek-r1-0528": "2025-05-28",
}

RELEASE_NOTES = (
    "Dates are public announcement dates for the specific checkpoint named by "
    "the source corpus. gpt3/chatgpt/gpt4 use the OpenAI snapshot IDs given in "
    "RAID Appendix E.2 (text-davinci-002, gpt-3.5-turbo-0613, gpt-4-0613). "
    "Base and chat variants of the same model share a date where they shipped "
    "together. Any date can be argued with; the mapping is in build_corpus.py "
    "so that argument is possible."
)


def _reservoir(bucket: list, item, cap: int, seen: int, rng: random.Random) -> None:
    """Standard reservoir sampling, so ordering in the source file cannot bias us."""
    if len(bucket) < cap:
        bucket.append(item)
    else:
        j = rng.randrange(seen)
        if j < cap:
            bucket[j] = item


def sample_raid(per_cell: int, seed: int, max_rows: int | None) -> list[dict]:
    """Stream RAID's labelled train split, reservoir-sampling per cell.

    Only unattacked generations are kept. RAID's twelve adversarial attacks are
    a separate axis and mixing them in would confound vintage with attack.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "liamdugan/raid", data_files={"train": "train.csv"}, split="train", streaming=True
    )
    rng = random.Random(seed)
    buckets: dict[tuple, list] = defaultdict(list)
    counts: dict[tuple, int] = defaultdict(int)

    for i, row in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        if (row.get("attack") or "none") != "none":
            continue
        text = (row.get("generation") or "").strip()
        if len(text.split()) < 80:
            continue
        model = (row.get("model") or "").strip()
        if model != "human" and model not in RELEASE:
            continue
        key = (model, row.get("domain"), row.get("decoding") or "na")
        counts[key] += 1
        _reservoir(
            buckets[key],
            {
                "text": text,
                "model": model,
                "domain": row.get("domain"),
                "decoding": row.get("decoding") or "na",
                "source_id": row.get("source_id"),
                "corpus": "RAID",
            },
            per_cell,
            counts[key],
            rng,
        )

    out = [r for b in buckets.values() for r in b]
    print(f"  RAID: {len(out)} documents across {len(buckets)} cells")
    return out


def sample_maga(per_cell: int, seed: int) -> list[dict]:
    """MAGA-Bench validation split, for generators released 2024 onward."""
    from huggingface_hub import hf_hub_download

    rng = random.Random(seed)
    buckets: dict[tuple, list] = defaultdict(list)
    counts: dict[tuple, int] = defaultdict(int)

    for fn in ("val/MAGA_val.jsonl", "val/MGB_val.jsonl"):
        try:
            path = hf_hub_download("anyangsong/MAGA", fn, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001
            print(f"  MAGA {fn}: unavailable ({type(exc).__name__})")
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(row.get("text") or row.get("content") or row.get("generation") or "").strip()
                if len(text.split()) < 80:
                    continue
                raw_model = str(row.get("model") or row.get("generator") or row.get("src") or "").strip().lower()
                label = row.get("label")
                is_human = raw_model in ("human", "") or label in (0, "0", "human")
                model = "human" if is_human else _normalise(raw_model)
                if model != "human" and model not in RELEASE:
                    continue
                key = (model, row.get("domain") or row.get("task") or "na", "na")
                counts[key] += 1
                _reservoir(
                    buckets[key],
                    {"text": text, "model": model, "domain": str(key[1]),
                     "decoding": "na", "source_id": row.get("id"), "corpus": "MAGA"},
                    per_cell, counts[key], rng,
                )

    out = [r for b in buckets.values() for r in b]
    print(f"  MAGA: {len(out)} documents across {len(buckets)} cells")
    return out


def _normalise(name: str) -> str:
    n = name.lower().replace("_", "-").strip()
    aliases = {
        "gpt4o-mini": "gpt-4o-mini", "gpt-4o mini": "gpt-4o-mini",
        "llama3.1-8b": "llama-3.1-8b", "llama-3.1-8b-instruct": "llama-3.1-8b",
        "gemini-2.0-flash-exp": "gemini-2.0-flash",
        "deepseek-v3-0324": "deepseek-v3", "deepseek-chat": "deepseek-v3",
        "qwen3-8b-instruct": "qwen3-8b", "qwen-plus": "qwen3-plus",
        "mistral-medium-3": "mistral-medium", "deepseek-r1": "deepseek-r1-0528",
    }
    return aliases.get(n, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="cap RAID rows streamed; 0 reads the whole split")
    args = ap.parse_args()

    print("sampling")
    rows = sample_raid(args.per_cell, args.seed, args.max_rows or None)
    rows += sample_maga(args.per_cell, args.seed)

    for r in rows:
        r["release_date"] = RELEASE.get(r["model"], "")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")

    from collections import Counter
    per_model = Counter(r["model"] for r in rows)
    print(f"\n{len(rows)} documents total")
    for model, n in sorted(per_model.items(), key=lambda kv: RELEASE.get(kv[0], "0000")):
        print(f"  {model:<20} {RELEASE.get(model, 'human'):<12} {n}")
    (OUT / "release_dates.json").write_text(
        json.dumps({"dates": RELEASE, "notes": RELEASE_NOTES}, indent=2)
    )
    print(f"\nwrote {OUT/'corpus.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
