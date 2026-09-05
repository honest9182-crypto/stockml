"""Up-only picker (step 1.6) -- see CLAUDE.md's "Up-only picker" section.

A picker is forced to name exactly `n_picks` tickers "up" every day; the
only decision is *which*. `scores.py` turns a model into a per-row `p_up`
score, `select.py` turns scores into picks, `baselines.py` provides the
mandatory comparison pickers, `evaluate.py` scores a picker's picks against
the day's own base rate, and `run.py` ties it all into a walk-forward run
matching step 1's shape.
"""
