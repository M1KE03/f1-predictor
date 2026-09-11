"""Single source of truth for column roles (spec section 3).

FEATURE_COLS is the exact list fed to the model. Everything else in
features.parquet is an identifier / label / helper and must never enter X.
"""

# Identifiers and raw-outcome columns kept in features.parquet for
# evaluation, auditing and joins -- NOT fed to the model.
ID_COLS = [
    "year", "round", "event_name", "circuit_id", "driver", "driver_id",
    "team", "date", "position", "status", "is_dnf", "classified", "points",
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
    # weather (race conditions)
    "air_temp", "track_temp", "humidity", "wind_speed", "rainfall", "is_wet",
    # weather affinity
    "driver_wet_delta", "driver_wet_n", "driver_temp_bin_avg", "driver_temp_bin_n",
    # circuit history
    "driver_circuit_avg_finish", "driver_circuit_best_finish",
    "driver_circuit_podium_rate", "driver_races_at_circuit", "is_rookie_here",
    "team_circuit_avg_finish",
    # teammate-relative
    "quali_gap_to_teammate", "form_finish_vs_teammate", "teammate_available",
    # reliability / car
    "driver_dnf_rate", "team_dnf_rate", "constructor_standing_prior",
]
