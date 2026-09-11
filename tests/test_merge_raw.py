"""Coverage checks when stitching separately-ingested season ranges together
(FIX_PLAN.md section 5.A.2)."""
from __future__ import annotations

import pandas as pd
import pytest

from src import merge_raw


def season(year: int, rounds: range, drivers: int = 20) -> pd.DataFrame:
    rows = []
    for rnd in rounds:
        date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(14 * rnd, unit="D")
        for d in range(drivers):
            rows.append(dict(year=year, round=rnd, date=date, driver=f"D{d:02d}"))
    return pd.DataFrame(rows)


def write(tmp_path, name: str, frame: pd.DataFrame):
    path = tmp_path / name
    frame.to_parquet(path, index=False)
    return path


def test_ranges_are_combined_in_date_order(tmp_path):
    a = write(tmp_path, "a.parquet", season(2019, range(1, 4)))
    b = write(tmp_path, "b.parquet", season(2022, range(1, 4)))
    combined = merge_raw.merge([b, a])          # deliberately out of order
    assert combined["date"].is_monotonic_increasing
    assert sorted(combined["year"].unique()) == [2019, 2022]


def test_an_overlapping_race_is_refused(tmp_path):
    """Ingesting the same race twice would double every historical aggregate
    built on top of it."""
    a = write(tmp_path, "a.parquet", season(2022, range(1, 4)))
    b = write(tmp_path, "b.parquet", season(2022, range(3, 6)))
    with pytest.raises(ValueError, match="more than one input"):
        merge_raw.merge([a, b])


def test_adjacent_ranges_do_not_trigger_the_overlap_check(tmp_path):
    a = write(tmp_path, "a.parquet", season(2022, range(1, 4)))
    b = write(tmp_path, "b.parquet", season(2022, range(4, 7)))
    assert merge_raw.merge([a, b]).groupby(["year", "round"]).ngroups == 6


def test_coverage_reports_counts_per_season(tmp_path):
    frame = merge_raw.merge([
        write(tmp_path, "a.parquet", season(2019, range(1, 22))),
        write(tmp_path, "b.parquet", season(2022, range(1, 23))),
    ])
    report = merge_raw.coverage_report(frame)
    assert report.loc[2019, "races"] == 21
    assert report.loc[2022, "races"] == 22
    assert report.loc[2019, "rows"] == 21 * 20


def test_a_hole_in_a_season_is_reported(tmp_path):
    """A season whose round numbers are not 1..max has races missing, which
    must be explained rather than silently accepted."""
    holed = pd.concat([season(2020, range(1, 5)), season(2020, range(7, 10))],
                      ignore_index=True)
    report = merge_raw.coverage_report(
        merge_raw.merge([write(tmp_path, "a.parquet", holed)]))
    assert report.loc[2020, "missing_rounds"] == [5, 6]


def test_a_complete_season_reports_no_gaps(tmp_path):
    report = merge_raw.coverage_report(
        merge_raw.merge([write(tmp_path, "a.parquet", season(2021, range(1, 23)))]))
    assert report.loc[2021, "missing_rounds"] == []


def test_variable_field_sizes_are_surfaced(tmp_path):
    """Field size is not fixed: the stored data holds 19-, 20- and 22-car races."""
    mixed = pd.concat([season(2026, range(1, 3), drivers=22),
                       season(2026, range(3, 4), drivers=19)], ignore_index=True)
    report = merge_raw.coverage_report(
        merge_raw.merge([write(tmp_path, "a.parquet", mixed)]))
    assert report.loc[2026, "min_field"] == 19
    assert report.loc[2026, "max_field"] == 22


# --- ingestion must not hide a systematic failure ---------------------------
# A real run fetched 8 of 17 rounds of 2020, skipped the rest on "any API:
# 500 calls/h", and exited 0. FIX_PLAN.md section 2 P1 flags exactly this.

import pytest as _pytest

from src import ingest


@_pytest.mark.parametrize("message", [
    "any API: 500 calls/h",
    "HTTP 429 Too Many Requests",
    "rate limit exceeded",
    "503 Service Unavailable",
    "Connection refused",
    "read timed out",
    "Temporarily unavailable",
])
def test_source_failures_are_systematic(message):
    assert ingest._is_systematic(Exception(message))


@_pytest.mark.parametrize("message", [
    "no data for this session",
    "empty results",
    "Session not available for this event",
    "The session does not exist",
])
def test_missing_sessions_are_not_systematic(message):
    """A genuinely absent session must still be skipped, not abort the run."""
    assert not ingest._is_systematic(Exception(message))


def test_detection_is_case_insensitive():
    assert ingest._is_systematic(Exception("ANY API: 500 CALLS/H"))


# --- hollow results ---------------------------------------------------------
# 2021 Qatar and 2026 Dutch entered the dataset with the right number of rows
# but blank status, blank classification and zero points across the field. The
# session loaded, race_rows() succeeded, and the run logged "OK ... rows=20"
# while writing a race in which everybody retired and nobody scored.

def test_a_hollow_payload_is_rejected():
    hollow = pd.DataFrame({"status": [""] * 20, "points": [0.0] * 20})
    with _pytest.raises(ValueError, match="hollow results"):
        ingest._reject_hollow_results(hollow, 2021, 20)


def test_a_normal_race_is_accepted():
    real = pd.DataFrame({"status": ["Finished"] * 10 + ["Retired"] * 10,
                         "points": [25.0, 18.0, 15.0] + [0.0] * 17})
    ingest._reject_hollow_results(real, 2021, 20)


def test_a_race_with_statuses_but_no_points_is_accepted():
    """Not hollow: the statuses prove a real payload. Guarding on points alone
    would reject a legitimately point-less field."""
    odd = pd.DataFrame({"status": ["Retired"] * 20, "points": [0.0] * 20})
    ingest._reject_hollow_results(odd, 2021, 20)


def test_a_race_with_points_but_blank_statuses_is_accepted():
    partial = pd.DataFrame({"status": [""] * 20,
                            "points": [25.0] + [0.0] * 19})
    ingest._reject_hollow_results(partial, 2021, 20)


def test_illness_is_a_retirement_not_a_non_start():
    """MAG 2020 Emilia Romagna: 47 laps of a 63-lap race, ClassifiedPosition
    'R'. He started and stopped."""
    from src import labels
    assert labels.normalize_status("Illness") == labels.RETIRED_UNSPECIFIED
    assert labels.normalize_status("Illness") not in (
        labels.DID_NOT_START, labels.WITHDRAWN, labels.UNKNOWN)
