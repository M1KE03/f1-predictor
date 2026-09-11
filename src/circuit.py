"""Circuit history features (spec 2.3). All stats use PRIOR races only."""
import pandas as pd

from .leakage import (assert_sorted, past_count, past_expanding_mean,
                      past_expanding_min, past_mean_excluding_current_race)


def add_circuit_history(df: pd.DataFrame) -> pd.DataFrame:
    assert_sorted(df)

    # position <= 3 is False (0.0) for NaN positions -- a DNF counts as
    # "not a podium", which is the intended semantics for a podium rate.
    podium = (df["position"] <= 3).astype(float)
    df = df.assign(_podium=podium)

    dc = ["driver", "circuit_id"]
    df["driver_circuit_avg_finish"] = past_expanding_mean(df, dc, "position")
    df["driver_circuit_best_finish"] = past_expanding_min(df, dc, "position")
    df["driver_circuit_podium_rate"] = past_expanding_mean(df, dc, "_podium")
    df["driver_races_at_circuit"] = past_count(df, dc).astype(float)
    df["is_rookie_here"] = (df["driver_races_at_circuit"] == 0).astype(int)

    # Team groups have TWO rows per race -> must exclude the whole current
    # race, not just the current row (see leakage.past_mean_excluding_current_race).
    df["team_circuit_avg_finish"] = past_mean_excluding_current_race(
        df, ["team", "circuit_id"], "position")

    return df.drop(columns=["_podium"])
