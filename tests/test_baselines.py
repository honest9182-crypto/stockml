"""Baselines produce the exact expected accuracy on a synthetic dataset with
a known class mix -- these are the numbers every other result is judged against.
"""

from __future__ import annotations

import pandas as pd

from stockml.models.baselines import AlwaysUp, MajorityClass


def test_majority_class_accuracy_matches_known_mix():
    # 50 "down", 30 "stagnant", 20 "up" -> majority is "down" -> train accuracy 0.5
    y = pd.Series(["down"] * 50 + ["stagnant"] * 30 + ["up"] * 20)
    X = pd.DataFrame(index=y.index)

    model = MajorityClass()
    model.fit(X, y)
    proba = model.predict_proba(X)
    preds = pd.Series(["down", "stagnant", "up"])[proba.argmax(axis=1)].reset_index(drop=True)

    accuracy = (preds.to_numpy() == y.to_numpy()).mean()
    assert accuracy == 0.5
    assert model._majority == "down"


def test_always_up_accuracy_matches_known_mix():
    # 20 "up" out of 100 -> always predicting "up" gives exactly 0.20 accuracy.
    y = pd.Series(["down"] * 50 + ["stagnant"] * 30 + ["up"] * 20)
    X = pd.DataFrame(index=y.index)

    model = AlwaysUp()
    model.fit(X, y)
    proba = model.predict_proba(X)
    preds = pd.Series(["down", "stagnant", "up"])[proba.argmax(axis=1)].reset_index(drop=True)

    accuracy = (preds.to_numpy() == y.to_numpy()).mean()
    assert accuracy == 0.20
    assert (preds == "up").all()


def test_always_up_ignores_training_data():
    y_all_down = pd.Series(["down"] * 10)
    model = AlwaysUp().fit(pd.DataFrame(index=y_all_down.index), y_all_down)
    proba = model.predict_proba(pd.DataFrame(index=range(5)))
    assert (proba.argmax(axis=1) == 2).all()  # index 2 == "up" in CLASS_ORDER
