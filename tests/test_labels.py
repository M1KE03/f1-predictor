"""Label-vocabulary fixtures (FIX_PLAN.md section 10).

Source-backed cases covering a normal race, a lapped runner, a classified
retirement, a bare retirement with no stated cause, DNS, DSQ and withdrawal.
The expected values are written out per row so that a change in
labels._STATUS_MAP has to be justified against a concrete result, not just
against the code.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import labels


# --- normalize_status -------------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    ("Finished", labels.FINISHED),
    ("Lapped", labels.FINISHED),
    ("+1 Lap", labels.FINISHED),
    ("+2 Laps", labels.FINISHED),
    ("+6 Laps", labels.FINISHED),
    ("Did not start", labels.DID_NOT_START),
    ("Withdrew", labels.WITHDRAWN),
    ("Disqualified", labels.DISQUALIFIED),
    ("Accident", labels.ACCIDENT),
    ("Collision", labels.ACCIDENT),
    ("Collision damage", labels.ACCIDENT),
    ("Spun off", labels.ACCIDENT),
    ("Engine", labels.MECHANICAL),
    ("Power Unit", labels.MECHANICAL),
    ("Hydraulics", labels.MECHANICAL),
    ("Gearbox", labels.MECHANICAL),
    ("Water pressure", labels.MECHANICAL),
])
def test_known_statuses_map_to_expected_category(status, expected):
    assert labels.normalize_status(status) == expected


def test_bare_retired_is_not_binned_as_mechanical():
    """'Retired' is 64.6% of retirements in the stored data and states no
    cause. Binning it as mechanical would invent a cause for two thirds of all
    retirements and corrupt any reliability feature built on it."""
    assert labels.normalize_status("Retired") == labels.RETIRED_UNSPECIFIED
    assert labels.normalize_status("Retired") != labels.MECHANICAL


@pytest.mark.parametrize("status", ["FINISHED", "  finished  ", "+1  Lap"])
def test_status_matching_is_case_and_whitespace_insensitive(status):
    assert labels.normalize_status(status) == labels.FINISHED


@pytest.mark.parametrize("status", [None, "", "   ", float("nan"), 42])
def test_missing_status_is_unknown(status):
    assert labels.normalize_status(status) == labels.UNKNOWN


def test_unmapped_status_warns_rather_than_guessing(caplog):
    with caplog.at_level("WARNING"):
        assert labels.normalize_status("Teleported") == labels.UNKNOWN
    assert "Teleported" in caplog.text


# --- officially_classified_from_raw -----------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("12", True), (5, True),
    ("R", False), ("D", False), ("E", False),
    ("W", False), ("F", False), ("N", False),
    ("r", False),
    (None, None), (float("nan"), None), ("", None), ("nan", None),
])
def test_classified_position_codes(raw, expected):
    assert labels.officially_classified_from_raw(raw) is expected


# --- derive_labels ----------------------------------------------------------

# Modelled on the 2026 round 1 classification, which contains a normal
# finisher, lapped runners, bare retirements and two DNS -- plus a DSQ and a
# withdrawal taken from other rounds of the stored dataset.
FIXTURE = pd.DataFrame([
    # driver, position, status
    ("WIN", 1.0, "Finished"),
    ("P03", 3.0, "Finished"),
    ("P10", 10.0, "Finished"),
    ("P11", 11.0, "Lapped"),
    ("LAP", 14.0, "+1 Lap"),
    ("RET", 18.0, "Retired"),
    ("ENG", 19.0, "Engine"),
    ("CRA", 20.0, "Accident"),
    ("DSQ", 21.0, "Disqualified"),
    ("DNS", 22.0, "Did not start"),
    ("WDR", float("nan"), "Withdrew"),
], columns=["driver", "position", "status"])


@pytest.fixture
def derived() -> pd.DataFrame:
    return labels.derive_labels(FIXTURE).set_index("driver")


def test_started_excludes_only_non_starters(derived):
    assert derived.loc["DNS", "started"] == 0
    assert derived.loc["WDR", "started"] == 0
    for driver in ["WIN", "P11", "RET", "ENG", "CRA", "DSQ"]:
        assert derived.loc[driver, "started"] == 1, driver


def test_finished_means_reached_the_flag(derived):
    for driver in ["WIN", "P03", "P10", "P11", "LAP"]:
        assert derived.loc[driver, "finished"] == 1, driver
    for driver in ["RET", "ENG", "CRA", "DSQ", "DNS", "WDR"]:
        assert derived.loc[driver, "finished"] == 0, driver


def test_lapped_runner_is_finished_not_retired(derived):
    """A lapped car reached the flag. The inherited code got this right only
    because it special-cased two spellings; the category makes it explicit."""
    assert derived.loc["LAP", "finished"] == 1
    assert derived.loc["LAP", "status_category"] == labels.FINISHED
    assert derived.loc["P11", "finished"] == 1


def test_disqualified_is_not_officially_classified(derived):
    assert derived.loc["DSQ", "officially_classified"] == 0
    assert derived.loc["DNS", "officially_classified"] == 0
    assert derived.loc["WDR", "officially_classified"] == 0


def test_classified_retirement_keeps_its_official_place(derived):
    """A retirement that still holds a place in the published order stays
    classified -- FIX_PLAN section 2 P0-6: do not collapse every retirement
    to last."""
    assert derived.loc["RET", "officially_classified"] == 1
    assert derived.loc["RET", "result_order"] == 18.0
    assert derived.loc["RET", "finished"] == 0


def test_result_order_is_preserved_not_collapsed(derived):
    assert derived.loc["WIN", "result_order"] == 1.0
    assert derived.loc["DSQ", "result_order"] == 21.0
    assert pd.isna(derived.loc["WDR", "result_order"])


def test_outcome_labels(derived):
    assert derived.loc["WIN", "is_winner"] == 1
    assert derived["is_winner"].sum() == 1
    assert derived.loc["P03", "is_podium"] == 1
    assert derived.loc["P10", "finished_top10"] == 1
    assert derived.loc["P11", "finished_top10"] == 0


def test_non_starters_and_dsq_never_get_a_top10_label(derived):
    """Verified against the stored dataset: FastF1 demotes DSQ and DNS to the
    back of the order, so no such row is labelled top-10."""
    for driver in ["DSQ", "DNS", "WDR"]:
        assert derived.loc[driver, "finished_top10"] == 0, driver
        assert derived.loc[driver, "is_podium"] == 0, driver
        assert derived.loc[driver, "is_winner"] == 0, driver


def test_classified_position_is_authoritative_when_present():
    """A late retirement that completed the distance IS officially classified.
    Status alone cannot know this; ClassifiedPosition can, and must win."""
    frame = pd.DataFrame([
        ("LATE", 11.0, "Retired", "11"),   # classified despite retiring
        ("EARLY", 20.0, "Retired", "R"),   # not classified
    ], columns=["driver", "position", "status", "ClassifiedPosition"])

    out = labels.derive_labels(
        frame, classified_position_col="ClassifiedPosition").set_index("driver")

    assert out.loc["LATE", "officially_classified"] == 1
    assert out.loc["EARLY", "officially_classified"] == 0
    assert (out["officially_classified_source"] == "classified_position").all()


def test_source_is_flagged_when_derived():
    out = labels.derive_labels(FIXTURE)
    assert (out["officially_classified_source"] == "derived_from_status").all()


def test_derive_labels_does_not_mutate_input():
    before = FIXTURE.copy()
    labels.derive_labels(FIXTURE)
    pd.testing.assert_frame_equal(FIXTURE, before)


def test_every_row_gets_a_known_category(derived):
    assert set(derived["status_category"]) <= set(labels.STATUS_CATEGORIES)
    assert labels.UNKNOWN not in set(derived["status_category"])
