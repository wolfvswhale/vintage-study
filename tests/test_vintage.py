import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyse import spearman, to_ordinal  # noqa: E402
from build_corpus import RELEASE, _normalise, _reservoir  # noqa: E402
from experiment import threshold_at_fpr  # noqa: E402


class TestSpearman:
    def test_perfect_positive(self):
        rho, n = spearman([1, 2, 3, 4], [10, 20, 30, 40])
        assert rho == pytest.approx(1.0)
        assert n == 4

    def test_perfect_negative(self):
        rho, _ = spearman([1, 2, 3, 4], [40, 30, 20, 10])
        assert rho == pytest.approx(-1.0)

    def test_monotonic_but_nonlinear(self):
        # Rank correlation should not care about the shape of the curve.
        rho, _ = spearman([1, 2, 3, 4], [1, 4, 9, 16])
        assert rho == pytest.approx(1.0)

    def test_ties_use_average_ranks(self):
        rho, _ = spearman([1, 2, 2, 3], [1, 2, 2, 3])
        assert rho == pytest.approx(1.0)

    def test_no_relationship_is_near_zero(self):
        rho, _ = spearman([1, 2, 3, 4, 5, 6], [3, 1, 4, 1, 5, 2])
        assert abs(rho) < 0.6

    def test_too_few_points_returns_nan(self):
        rho, n = spearman([1, 2], [3, 4])
        assert np.isnan(rho) and n == 2

    def test_matches_scipy(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        xs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        ys = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0]
        assert spearman(xs, ys)[0] == pytest.approx(scipy_stats.spearmanr(xs, ys).statistic)


class TestThreshold:
    def test_fixes_false_positive_rate(self):
        scores = np.linspace(0, 1, 1000)
        thr = threshold_at_fpr(scores, 0.05)
        assert (scores >= thr).mean() == pytest.approx(0.05, abs=0.01)

    def test_stricter_target_raises_threshold(self):
        scores = np.linspace(0, 1, 1000)
        assert threshold_at_fpr(scores, 0.01) > threshold_at_fpr(scores, 0.10)

    def test_handles_constant_scores(self):
        assert threshold_at_fpr(np.full(100, 0.5), 0.05) == pytest.approx(0.5)


class TestReservoir:
    def test_fills_up_to_cap(self):
        import random

        rng = random.Random(0)
        bucket = []
        for i in range(5):
            _reservoir(bucket, i, 10, i + 1, rng)
        assert bucket == list(range(5))

    def test_never_exceeds_cap(self):
        import random

        rng = random.Random(0)
        bucket = []
        for i in range(1000):
            _reservoir(bucket, i, 25, i + 1, rng)
        assert len(bucket) == 25

    def test_samples_beyond_the_first_cap_items(self):
        """The point of reservoir sampling: ordering in the source must not bias us."""
        import random

        rng = random.Random(7)
        bucket = []
        for i in range(2000):
            _reservoir(bucket, i, 20, i + 1, rng)
        assert max(bucket) > 100, "sample is stuck at the head of the stream"


class TestReleaseDates:
    def test_every_date_parses(self):
        for model, d in RELEASE.items():
            assert to_ordinal(d) > 0, model

    def test_dates_are_ordered_sensibly(self):
        assert to_ordinal(RELEASE["gpt2"]) < to_ordinal(RELEASE["gpt4"])
        assert to_ordinal(RELEASE["gpt4"]) < to_ordinal(RELEASE["gpt-4o-mini"])
        assert to_ordinal(RELEASE["gpt-4o-mini"]) < to_ordinal(RELEASE["qwen3-8b"])

    def test_base_and_chat_variants_share_a_date(self):
        """The paired comparison depends on this being true."""
        for base, chat in (("mpt", "mpt-chat"), ("mistral", "mistral-chat")):
            assert RELEASE[base] == RELEASE[chat]

    def test_normalise_maps_aliases(self):
        assert _normalise("DeepSeek-Chat") == "deepseek-v3"
        assert _normalise("Llama-3.1-8B-Instruct") == "llama-3.1-8b"
        assert _normalise("qwen3-8b") == "qwen3-8b"


@pytest.fixture(scope="module")
def results():
    p = Path(__file__).resolve().parents[1] / "results" / "vintage.json"
    if not p.exists():
        pytest.skip("results not built")
    return json.loads(p.read_text())


class TestResultsIntegrity:
    """Guards on the shipped results, so a bad rerun cannot quietly replace them."""

    def test_every_detector_hits_its_target_fpr(self, results):
        for name, d in results["detectors"].items():
            assert d["achieved_fpr"] == pytest.approx(0.05, abs=0.015), name

    def test_training_generators_all_predate_the_cutoff(self, results):
        cutoff = results["train_before"]
        pg = next(iter(results["detectors"].values()))["per_generator"]
        for m in results["train_models"]:
            assert pg[m]["release_date"] < cutoff, m

    def test_recalls_are_probabilities(self, results):
        for d in results["detectors"].values():
            for m, g in d["per_generator"].items():
                assert 0.0 <= g["recall_at_5pct_fpr"] <= 1.0, m

    def test_every_generator_has_a_release_date(self, results):
        for d in results["detectors"].values():
            for m, g in d["per_generator"].items():
                assert g["release_date"], m
