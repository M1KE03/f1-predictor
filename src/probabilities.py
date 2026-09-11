"""Race-level probabilities from ranker scores (FIX_PLAN.md section 6).

Everything so far produces an ORDER. An order cannot answer "how likely is
Verstappen to win?", cannot be scored by log loss or Brier, and cannot express
that one race is a near-certainty while another is a coin toss. Two of the
promotion gates have therefore never been scored at all.

The model
---------
A Plackett-Luce ranking distribution. Each driver has a utility `s_i` (the
ranker's score); the probability of finishing first is a softmax

    P(win_i) = exp(s_i / T) / sum_j exp(s_j / T)

and subsequent positions are drawn from the same form without replacement. One
coherent distribution over orderings yields win, podium and top-10
probabilities together, so they cannot contradict each other -- unlike
normalising three independent classifier outputs, which FIX_PLAN.md section 6
warns against explicitly.

Sampling is the Gumbel-max trick: adding independent Gumbel(0,1) noise to the
scaled utilities and sorting gives an EXACT draw from the Plackett-Luce
distribution, not an approximation of one.

Temperature
-----------
`T` controls confidence without changing the order, so calibration alone can
never fix a wrong winner pick -- it can only stop the model being overconfident
about it. It is fitted on chronological out-of-sample development predictions
(Guo et al., temperature scaling), never on the block being scored.

Caveat worth keeping in view: mapping ranker scores through a softmax is a
DISTRIBUTIONAL ASSUMPTION, not automatic calibration. The coherence checks
below verify internal consistency; only the log-loss and Brier gates test
whether the assumption is any good.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

RACE_KEYS: Final = ["year", "round"]

DEFAULT_DRAWS: Final = 20_000
DEFAULT_SEED: Final = 20260911

# Temperature search grid. Log-spaced: what matters is the ratio, not the step.
TEMPERATURE_GRID: Final = tuple(np.exp(np.linspace(np.log(0.05), np.log(20.0), 60)))

PODIUM_SIZE: Final = 3
TOP10_SIZE: Final = 10


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable softmax of `scores / temperature`."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}.")
    scaled = np.asarray(scores, dtype=float) / temperature
    scaled -= scaled.max()
    weights = np.exp(scaled)
    return weights / weights.sum()


def sample_orderings(scores: np.ndarray, temperature: float,
                     n_draws: int = DEFAULT_DRAWS,
                     seed: int = DEFAULT_SEED) -> np.ndarray:
    """Draw orderings from the Plackett-Luce distribution.

    Returns an array of shape (n_draws, n_drivers) holding driver INDICES in
    finishing order. Uses Gumbel-max, which is exact for Plackett-Luce.
    """
    scaled = np.asarray(scores, dtype=float) / temperature
    rng = np.random.default_rng(seed)
    gumbel = rng.gumbel(size=(n_draws, scaled.size))
    return np.argsort(-(scaled + gumbel), axis=1)


def race_probabilities(scores: np.ndarray, temperature: float,
                       n_draws: int = DEFAULT_DRAWS,
                       seed: int = DEFAULT_SEED) -> dict[str, np.ndarray]:
    """Win / podium / top-10 probabilities for one race.

    `p_win` is computed in closed form rather than sampled: it is exactly the
    softmax, so sampling it would only add Monte Carlo error to a known answer.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    p_win = softmax(scores, temperature)

    orderings = sample_orderings(scores, temperature, n_draws, seed)
    podium_size = min(PODIUM_SIZE, n)
    top10_size = min(TOP10_SIZE, n)

    p_podium = np.zeros(n)
    p_top10 = np.zeros(n)
    np.add.at(p_podium, orderings[:, :podium_size].ravel(), 1.0)
    np.add.at(p_top10, orderings[:, :top10_size].ravel(), 1.0)
    p_podium /= n_draws
    p_top10 /= n_draws

    # Monte Carlo standard error on a proportion, so the caller can tell
    # whether a difference between two drivers is resolved by the draw count.
    mc_error = float(np.sqrt(0.25 / n_draws))
    return {"p_win": p_win, "p_podium": p_podium, "p_top10": p_top10,
            "mc_error": mc_error}


def add_probabilities(df: pd.DataFrame, score_col: str, temperature: float,
                      n_draws: int = DEFAULT_DRAWS,
                      seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Attach p_win / p_podium / p_top10 to every row, race by race."""
    out = df.copy()
    for column in ("p_win", "p_podium", "p_top10"):
        out[column] = np.nan

    for i, (_, race) in enumerate(out.groupby(RACE_KEYS, sort=False)):
        probs = race_probabilities(race[score_col].to_numpy(), temperature,
                                   n_draws, seed + i)
        for column in ("p_win", "p_podium", "p_top10"):
            out.loc[race.index, column] = probs[column]
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def winner_log_loss(df: pd.DataFrame, prob_col: str = "p_win",
                    winner_col: str = "is_winner") -> float:
    """Mean -log(probability assigned to the actual winner), per race.

    One value per RACE, not per driver: a race is one observation.
    """
    losses = []
    for _, race in df.groupby(RACE_KEYS, sort=False):
        winner = race[race[winner_col] == 1]
        if not len(winner):
            continue
        p = float(winner[prob_col].iloc[0])
        losses.append(-np.log(max(p, 1e-15)))
    return float(np.mean(losses)) if losses else float("nan")


def fit_temperature(df: pd.DataFrame, score_col: str,
                    grid: tuple[float, ...] = TEMPERATURE_GRID) -> float:
    """Choose T minimising winner log loss on DEVELOPMENT rows.

    Never call this on the block being scored. Temperature changes confidence
    without changing order, so fitting it on the test block would flatter the
    probability metrics while leaving every ranking metric untouched -- an easy
    leak to miss.
    """
    best_t, best_loss = float(grid[len(grid) // 2]), float("inf")
    for temperature in grid:
        scored = df.copy()
        probs = []
        for _, race in scored.groupby(RACE_KEYS, sort=False):
            probs.append(pd.Series(softmax(race[score_col].to_numpy(), temperature),
                                   index=race.index))
        scored["p_win"] = pd.concat(probs)
        loss = winner_log_loss(scored)
        if np.isfinite(loss) and loss < best_loss:
            best_loss, best_t = loss, float(temperature)
    return best_t


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------
def coherence_report(df: pd.DataFrame, tolerance: float = 0.02) -> dict[str, object]:
    """Check the totals a coherent ranking distribution must satisfy.

    FIX_PLAN.md section 6: sum(p_win) = 1, sum(p_podium) = 3, sum(p_top10) = 10,
    and p_win <= p_podium <= p_top10 for every driver. Independent binary
    outputs do NOT satisfy these automatically, which is the point of deriving
    all three from one distribution.
    """
    problems: list[str] = []
    n_races = 0
    for (year, rnd), race in df.groupby(RACE_KEYS, sort=False):
        n_races += 1
        n = len(race)
        for column, expected in (("p_win", 1.0),
                                 ("p_podium", float(min(PODIUM_SIZE, n))),
                                 ("p_top10", float(min(TOP10_SIZE, n)))):
            total = float(race[column].sum())
            if abs(total - expected) > tolerance:
                problems.append(
                    f"{year} R{rnd}: sum({column}) = {total:.4f}, expected {expected}")
        if not (race["p_win"] <= race["p_podium"] + tolerance).all():
            problems.append(f"{year} R{rnd}: p_win exceeds p_podium")
        if not (race["p_podium"] <= race["p_top10"] + tolerance).all():
            problems.append(f"{year} R{rnd}: p_podium exceeds p_top10")

    return {"n_races": n_races, "n_problems": len(problems),
            "coherent": not problems, "problems": problems[:20]}


def podium_brier(df: pd.DataFrame) -> float:
    """Per-driver squared error on the podium indicator, averaged within each
    race and then across races, so a race is one observation."""
    per_race = []
    for _, race in df.groupby(RACE_KEYS, sort=False):
        error = (race["p_podium"] - race["is_podium"]) ** 2
        per_race.append(float(error.mean()))
    return float(np.mean(per_race)) if per_race else float("nan")


def uniform_winner_log_loss(df: pd.DataFrame) -> float:
    """Log loss of a model that says every entrant is equally likely.

    The sanity floor FIX_PLAN.md section 8 asks for: a probability model that
    cannot beat this has learned nothing at all.
    """
    losses = [np.log(len(race)) for _, race in df.groupby(RACE_KEYS, sort=False)
              if (race["is_winner"] == 1).any()]
    return float(np.mean(losses)) if losses else float("nan")
