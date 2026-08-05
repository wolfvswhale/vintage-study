"""Does detector accuracy fall as the generator gets newer?

The design is deliberately plain, because the point is the curve, not the
detector.

Train on the oldest generators only, simulating someone who built a detector
when those were the newest models available. Then measure per-generator recall
against every generator in the corpus, ordered by public release date.

Two decisions make the numbers comparable across generators.

**The threshold is fixed by the human class, not by accuracy.** For each
detector the score threshold is set so that exactly 5% of held-out human
documents are flagged. Per-generator recall is then measured at that fixed
false-positive rate. Reporting raw accuracy instead would let a detector look
better on a generator simply by becoming more trigger-happy.

**Human documents are never split by vintage,** because they have none. The
same held-out human pool is used for every generator, so any movement in the
curve comes from the machine side.

    python experiment.py --train-before 2023-07-01
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prose-eval"))
sys.path.insert(0, str(HERE.parent / "bluepencil"))

TARGET_FPR = 0.05


def load_corpus(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_detectors():
    """Three detectors with different inductive biases.

    If the vintage curve is real it should appear in all three. If it appears in
    only one it is a property of that model, not of the text.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    def char_tfidf():
        return Pipeline([
            ("v", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                  min_df=3, max_features=50000, lowercase=False)),
            ("c", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])

    def word_tfidf():
        return Pipeline([
            ("v", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                  min_df=3, max_features=50000)),
            ("c", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])

    return {"char tf-idf": char_tfidf, "word tf-idf": word_tfidf}


class StyleDetector:
    """Interpretable features from prose-eval plus the bluepencil gate statistics.

    Included because it is the one detector here whose features can be named,
    which makes a per-feature account of any degradation possible.
    """

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.pipe = Pipeline([
            ("s", StandardScaler()),
            ("c", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ])

    @staticmethod
    def _features(texts):
        from proseeval import features as pe

        return pe.to_matrix(pe.extract_many(list(texts)))

    def fit(self, texts, y):
        self.pipe.fit(self._features(texts), y)
        return self

    def predict_proba(self, texts):
        return self.pipe.predict_proba(self._features(texts))


def threshold_at_fpr(human_scores: np.ndarray, target: float = TARGET_FPR) -> float:
    """Score above which exactly `target` of human documents fall."""
    return float(np.quantile(human_scores, 1 - target))


def run(corpus, train_before: str, seed: int, min_docs: int) -> dict:
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    humans = [r for r in corpus if r["model"] == "human"]
    machines = [r for r in corpus if r["model"] != "human" and r.get("release_date")]

    by_model = defaultdict(list)
    for r in machines:
        by_model[r["model"]].append(r)
    by_model = {m: rows for m, rows in by_model.items() if len(rows) >= min_docs}

    train_models = sorted(m for m in by_model if by_model[m][0]["release_date"] < train_before)
    if not train_models:
        raise SystemExit(f"no generators released before {train_before}")

    # Human pool split once and shared by every evaluation.
    h_train, h_test = train_test_split(humans, test_size=0.4, random_state=seed)

    # Training machine documents come only from the old generators.
    m_train = [r for m in train_models for r in by_model[m]]
    # Hold out a slice of each training generator so in-distribution recall is
    # measured on unseen documents rather than on the training set.
    held = {}
    m_train_final = []
    for m in train_models:
        rows = by_model[m][:]
        rng.shuffle(rows)
        # 30% held out, so in-distribution recall is measured on a slice large
        # enough to compare against the out-of-distribution generators rather
        # than on a handful of documents.
        cut = max(int(0.3 * len(rows)), min_docs)
        held[m] = rows[:cut]
        m_train_final += rows[cut:]

    train_texts = [r["text"] for r in h_train] + [r["text"] for r in m_train_final]
    train_y = np.array([0] * len(h_train) + [1] * len(m_train_final))
    human_test_texts = [r["text"] for r in h_test]

    print(f"training on {len(train_models)} generators released before {train_before}: "
          f"{', '.join(train_models)}")
    print(f"  {len(m_train_final)} machine documents, {len(h_train)} human")
    print(f"  held-out human for thresholding: {len(h_test)}\n")

    detectors = build_detectors()
    results = {}

    for name, factory in list(detectors.items()) + [("style features", None)]:
        model = StyleDetector() if factory is None else factory()
        model.fit(train_texts, train_y)
        human_scores = model.predict_proba(human_test_texts)[:, 1]
        thr = threshold_at_fpr(human_scores)
        achieved_fpr = float((human_scores >= thr).mean())

        per_gen = {}
        for m, rows in sorted(by_model.items(), key=lambda kv: kv[1][0]["release_date"]):
            eval_rows = held[m] if m in held else rows
            scores = model.predict_proba([r["text"] for r in eval_rows])[:, 1]
            per_gen[m] = {
                "release_date": rows[0]["release_date"],
                "corpus": rows[0]["corpus"],
                "n": len(eval_rows),
                "recall_at_5pct_fpr": float((scores >= thr).mean()),
                "mean_score": float(scores.mean()),
                "in_training": m in train_models,
            }
        results[name] = {"threshold": thr, "achieved_fpr": achieved_fpr, "per_generator": per_gen}
        print(f"{name}: threshold {thr:.3f}, human FPR {achieved_fpr:.1%}")

    return {"train_models": train_models, "train_before": train_before, "detectors": results}


def report(out: dict) -> None:
    names = list(out["detectors"])
    first = out["detectors"][names[0]]["per_generator"]
    order = sorted(first, key=lambda m: (first[m]["release_date"], m))

    header = f"{'generator':<20} {'released':<12} {'corpus':<6} {'n':>5} " + "".join(f"{n:>16}" for n in names)
    print("\n" + header)
    print("-" * len(header))
    for m in order:
        row = f"{m:<20} {first[m]['release_date']:<12} {first[m]['corpus']:<6} {first[m]['n']:>5} "
        for n in names:
            g = out["detectors"][n]["per_generator"][m]
            mark = "*" if g["in_training"] else " "
            row += f"{g['recall_at_5pct_fpr']:>15.1%}{mark}"
        print(row)
    print("\n* generator was in the detector's training set")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "data" / "corpus.jsonl"))
    ap.add_argument("--train-before", default="2023-07-01",
                    help="train only on generators released before this date")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-docs", type=int, default=30)
    ap.add_argument("--out", default=str(HERE / "results" / "vintage.json"))
    args = ap.parse_args()

    corpus = load_corpus(Path(args.corpus))
    print(f"{len(corpus)} documents loaded\n")
    out = run(corpus, args.train_before, args.seed, args.min_docs)
    report(out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
