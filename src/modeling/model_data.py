"""Shared cohort definitions for the modelling workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.build_cohort import load_clean_cohort


SPLITS = {
    "FY2010+": {
        "fold1_train": (2010, 2012),
        "fold1_validate": (2013, 2013),
        "fold2_train": (2010, 2013),
        "fold2_validate": (2014, 2014),
        "final_refit": (2010, 2014),
        "locked_test": (2015, 2016),
    },
    "FY2001+": {
        "fold1_train": (2001, 2009),
        "fold1_validate": (2010, 2011),
        "fold2_train": (2001, 2011),
        "fold2_validate": (2012, 2014),
        "final_refit": (2001, 2014),
        "locked_test": (2015, 2016),
    },
}


def load_cohort(
    raw_dir: Path, chunk_size: int = 100_000
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the cleaned FY2001+ cohort used by the model experiments."""
    return load_clean_cohort(raw_dir, chunk_size)


def population(cohort: pd.DataFrame, design: str) -> pd.DataFrame:
    """Return the population required by one experiment design."""
    if design not in SPLITS:
        raise ValueError(f"Unknown modelling design: {design}")

    if design == "FY2001+":
        return cohort.copy()

    valid_interest = cohort["initial_interest_rate"].notna() & cohort[
        "initial_interest_rate"
    ].ge(0)
    return cohort[cohort["approval_fy"].ge(2010) & valid_interest].copy()


def years(data: pd.DataFrame, span: tuple[int, int]) -> pd.DataFrame:
    """Select an inclusive fiscal-year range."""
    start_year, end_year = span
    return data[data["approval_fy"].between(start_year, end_year)].copy()


def features(
    design: str,
    state: bool = False,
    processing: bool = False,
    interest: bool = True,
) -> list[str]:
    """Return the feature list for a controlled model variant."""
    if design not in SPLITS:
        raise ValueError(f"Unknown modelling design: {design}")

    selected = [
        "gross_approval",
        "term_months",
        "guarantee_percent",
        "naics2",
        "business_age",
        "business_type",
    ]
    if state:
        selected.append("project_state")
    if processing:
        selected.append("processing_method")
    if design == "FY2010+" and interest:
        selected.append("initial_interest_rate")
    return selected
