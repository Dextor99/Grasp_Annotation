import numpy as np

from scripts.common_eval.summarize_stratified_subset import summarize_stratified_subset


def test_weighted_stratified_rates_use_population_weights():
    # Two grasp points, four geometry candidates each. Point 0 has three
    # geometry-valid candidates and point 1 has one, so point-balanced and
    # candidate-weighted rates intentionally differ.
    collision = np.array([
        [[[False, False, False, True]]],
        [[[False, True, True, True]]],
    ])
    candidate_ids = np.array([0, 1, 4], dtype=np.int64)
    # point 0: scores 0.2 (FC/HQ), -1 (FC-invalid), point 1: 0.8 (FC)
    scores = np.array([0.2, -1.0, 0.8], dtype=float)

    summary, rows = summarize_stratified_subset(
        collision=collision,
        candidate_ids=candidate_ids,
        scores=scores,
        hq_threshold=0.4,
    )

    assert summary["population_geometry_valid"] == 4
    assert summary["sample_count"] == 3
    assert summary["weighted_fc_rate"] == 0.625
    assert summary["weighted_hq_probability"] == 0.375
    assert summary["weighted_hq_rate_among_fc"] == 0.6
    assert summary["unweighted_fc_rate"] == 2 / 3
    assert summary["unweighted_hq_probability"] == 1 / 3
    assert np.isclose(summary["weighted_mean_mu"], 0.44)
    assert [row["population_size"] for row in rows] == [3, 1]
    assert [row["sample_size"] for row in rows] == [2, 1]
