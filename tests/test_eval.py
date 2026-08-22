import pandas as pd

from relplatform.eval.root_cause_eval import build_hand_label_sample, evaluate_keyword_baseline, keyword_baseline_predict


def test_keyword_baseline_predicts_something_valid(sim_result):
    incidents = pd.DataFrame(sim_result.incidents)
    from relplatform.generator.postmortems import CATEGORIES

    for text in incidents["postmortem_text"].head(20):
        assert keyword_baseline_predict(text) in CATEGORIES


def test_hand_label_sample_is_stratified_and_bounded(sim_result):
    incidents = pd.DataFrame(sim_result.incidents)
    sample = build_hand_label_sample(incidents, n=50)
    assert len(sample) <= 50
    assert sample["id"].is_unique
    # more than one category should be represented if the underlying data has variety
    assert sample["root_cause_category"].nunique() > 1


def test_keyword_baseline_beats_random_guessing(sim_result):
    incidents = pd.DataFrame(sim_result.incidents)
    sample = build_hand_label_sample(incidents, n=80)
    result = evaluate_keyword_baseline(sample)
    n_categories = sample["root_cause_category"].nunique()
    assert result["accuracy"] > 1.0 / n_categories
