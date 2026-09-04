"""model_builder.build_model: every model_family produces a working
fit/predict_proba model; "xgb" specifically maps genome.hgb_* fields onto
XGBoost's own parameter names (models/xgb_model.py's documented mapping) --
it's a new gene *mapping*, not a new gene set, so this is what actually
proves it's wired correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockml.evolution.genome import Genome
from stockml.evolution.model_builder import build_model
from stockml.features import feature_names


def _tiny_xy(n: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    feat_cols = feature_names()
    X = pd.DataFrame(rng.normal(size=(n, len(feat_cols))), columns=feat_cols)
    y = pd.Series(rng.choice(["down", "stagnant", "up"], size=n))
    return X, y


@pytest.mark.parametrize("model_family", ["logreg", "hgb", "xgb"])
def test_build_model_fits_and_predicts_valid_proba(model_family):
    genome = Genome(model_family=model_family)
    model = build_model(genome, seed=0)
    X, y = _tiny_xy()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all()


def test_xgb_maps_hgb_genes_onto_xgboost_params():
    # Values must be actual GENE_GRIDS members -- Genome.__post_init__
    # rejects anything else, same as it would for a real evolved genome.
    genome = Genome(
        model_family="xgb", hgb_max_depth=4, hgb_learning_rate=0.3,
        hgb_max_iter=200, hgb_min_samples_leaf=100, hgb_l2=1.0,
    )
    model = build_model(genome, seed=0)
    xgb_params = model._base._clf.get_params()  # BiasedModel -> XGB -> xgboost.XGBClassifier
    assert xgb_params["max_depth"] == 4
    assert xgb_params["learning_rate"] == 0.3
    assert xgb_params["n_estimators"] == 200
    assert xgb_params["min_child_weight"] == 100
    assert xgb_params["reg_lambda"] == 1.0
    assert xgb_params["tree_method"] == "hist"
    assert xgb_params["device"] in ("cpu", "cuda")


def test_unknown_model_family_raises():
    # Genome.__post_init__ already rejects an invalid model_family at
    # construction (test_genome.py) -- this is build_model's own defensive
    # branch, reachable only by bypassing that validation (e.g. a forward-
    # compatibility scenario: a family newer code wrote that this code
    # doesn't know how to build).
    genome = Genome()
    object.__setattr__(genome, "model_family", "not_a_family")
    with pytest.raises(ValueError):
        build_model(genome, seed=0)
