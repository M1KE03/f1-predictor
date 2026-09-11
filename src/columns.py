"""Single source of truth for column roles (spec section 3).

FEATURE_COLS is the exact list fed to the model. Everything else in
features.parquet is an identifier / label / helper and must never enter X.
"""

# Identifiers and raw-outcome columns kept in features.parquet for
# evaluation, auditing and joins -- NOT fed to the model.
ID_COLS = [
    "year", "round", "event_name", "circuit_id", "driver", "driver_id",
    "team", "date", "position", "status", "points",
    # Separated outcome concepts (src.labels). `position` above is the same
    # values as `result_order`; both are kept while callers migrate.
    "result_order", "classified_position_raw", "status_category",
    "started", "finished", "officially_classified",
    "officially_classified_source", "is_winner", "is_podium",
    # DEPRECATED. `classified` means only "a result place exists" and is true
    # for 99.9% of rows -- use `officially_classified`. `is_dnf` is the
    # complement of `finished`. Retained so unmigrated callers keep working;
    # remove once nothing reads them.
    "is_dnf", "classified",
    # helper kept for the fill policy + audit (spec 2.2 computes it as an
    # intermediate; spec 3 deliberately excludes it from FEATURE_COLS)
    "driver_overall_avg_finish",
]

TARGET = "finished_top10"

FEATURE_COLS = [
    # grid
    "grid_position", "pit_start",
    # recent form
    "form_avg_finish_3", "form_avg_points_3", "form_avg_quali_3",
    "form_dnf_rate_5", "season_avg_finish", "momentum",
    # Weather affinity -- driver traits computed from PRIOR races only, so they
    # are known at the forecast cutoff. See WEATHER_REMOVED below for the eight
    # weather inputs that were removed because they are not.
    "driver_wet_delta", "driver_wet_n",
    # circuit history
    "driver_circuit_avg_finish", "driver_circuit_best_finish",
    "driver_circuit_podium_rate", "driver_races_at_circuit", "is_rookie_here",
    "team_circuit_avg_finish",
    # teammate-relative
    "quali_gap_to_teammate", "form_finish_vs_teammate", "teammate_available",
    # reliability / car
    "driver_dnf_rate", "team_dnf_rate", "constructor_standing_prior",
    # Current-weekend qualifying pace (src/qualifying.py). Measured at the
    # qualifying session, which ends before the forecast cutoff. Gaps are
    # normalised WITHIN a segment, because the minimum across Q1/Q2/Q3 compares
    # a Q1-eliminated driver's lap against another driver's Q3 lap.
    "quali_gap_pct", "quali_gap_to_median_pct", "quali_pace_vs_teammate_pct",
    "quali_stage_reached", "quali_field_spread_pct", "quali_no_time",
    "q1_gap_pct",
    # Recency-weighted race pace (src/ratings.py). The first features that
    # measure how FAST a car is over a stint rather than where it started.
    # Aggregated over PRIOR races only.
    "driver_pace_ewm_3", "driver_pace_ewm_12", "driver_pace_trend",
    "team_pace_ewm_3", "team_pace_ewm_12", "team_pace_trend",
    "driver_pace_vs_team", "driver_pace_sd_ewm", "driver_pace_races",
]

# ---------------------------------------------------------------------------
# Removed from FEATURE_COLS by increment 1.4 (FIX_PLAN.md section 2, P0-1).
#
# Two distinct leaks, both of which made the model depend on information that
# does not exist before lights-out:
#
# 1. The six raw weather readings are produced by ingest.weather_row(), which
#    averages FastF1's weather stream over the RACE SESSION. A forecast made an
#    hour before the start cannot know the race's mean air temperature, so
#    training on it measures hindsight, not prediction.
#
# 2. driver_temp_bin_* passed the "prior races only" test but still leaked:
#    weather.add_weather_affinity() chose WHICH historical temperature bucket to
#    read using the target race's realized track temperature. The values were
#    historical; the selection was not.
#
# The kept driver_wet_* features are genuinely clean -- they count and average a
# driver's PRIOR wet races, and their value at a target row is unchanged by that
# race's own conditions.
#
# These return only with a forecast whose pre-cutoff availability can be proven
# (FIX_PLAN.md section 5.D), not with observed or reanalysis weather.
# ---------------------------------------------------------------------------
WEATHER_REMOVED = [
    "air_temp", "track_temp", "humidity", "wind_speed", "rainfall", "is_wet",
    "driver_temp_bin_avg", "driver_temp_bin_n",
]

# Raw weather stays in features.parquet as an identifier/helper column: it is
# needed to build the historical wet-affinity subsets and to audit this change.
# Being in ID_COLS is what guarantees it never reaches the model.
ID_COLS += ["air_temp", "track_temp", "humidity", "wind_speed", "rainfall", "is_wet"]
# Raw qualifying times: kept for auditing the normalised gaps built from them.
# quali_best_s is the min across segments and is NOT comparable between drivers
# -- it stays out of the model deliberately.
ID_COLS += ["q1_s", "q2_s", "q3_s", "quali_best_s", "gap_to_pole_s"]
# Raw race pace: an OUTCOME of the race it describes, so it is an identifier
# here and only ever reaches the model through src/ratings.py history.
ID_COLS += ["race_pace_pct", "race_pace_laps", "race_pace_sd_pct"]

# ---------------------------------------------------------------------------
# Practice long-run pace: INGESTED AND MEASURED, but NOT a model input.
#
# FIX_PLAN.md section 5.D required these to be kept "only if they improve
# chronological validation". They did not -- they made things significantly
# worse. On the same 127 backtested races, adding them moved ranker winner
# accuracy 0.6063 -> 0.5276, a paired -0.0787 [-0.1417, -0.0157] whose interval
# EXCLUDES ZERO, and winner log loss 1.1469 -> 1.1861.
#
# The diagnosis is in the correlations: practice pace correlates only 0.215
# with finishing order (against 0.618 for grid position) because fuel loads and
# run plans are not observable, so a heavy-fuel race simulation and a low-fuel
# run are indistinguishable. It IS largely independent of grid (0.252), which
# is why it was worth trying -- but weak and independent is still weak, and
# five noisy columns on ~3000 training rows displace better signal.
#
# The data and src/practice.py are retained: the measurement is sound, and the
# features may become usable with stint-level fuel correction or many more
# seasons. Do not re-add them to FEATURE_COLS without re-running the backtest.
# ---------------------------------------------------------------------------
ID_COLS += ["practice_pace_pct", "practice_laps", "practice_sd_pct",
            "practice_deg_slope", "practice_runs"]

assert not (set(FEATURE_COLS) & set(WEATHER_REMOVED)), \
    "Race-session weather must not re-enter FEATURE_COLS without a proven forecast source."
