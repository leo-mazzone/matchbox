"""Direct unit tests for `SplinkLinker` behavior.

These tests cover Splink-specific schema handling and diagnostics, rather than the
shared match/no-match behavior exercised in the other linker suites.
"""

from datetime import date
from unittest.mock import Mock, patch

import polars as pl
from splink import SettingsCreator
from splink import blocking_rule_library as brl
from splink import comparison_library as cl

from matchlab.models.linkers.splinklinker import SplinkLinker


def test_splink_schema_order() -> None:
    """Splink accepts matching schemas even when the incoming column order differs."""
    linker = SplinkLinker(
        left_id="id",
        right_id="id",
        linker_training_functions=[],
        linker_settings=SettingsCreator(
            link_type="link_only",
            blocking_rules_to_generate_predictions=[brl.block_on("company_name")],
            comparisons=[cl.ExactMatch("company_name")],
        ),
        threshold=None,
    )
    left = pl.DataFrame(
        {
            "id": [1],
            "postcode": ["SW1A 1AA"],
            "duns_number": ["123456789"],
            "company_name": ["left"],
            "incorporation_date": [date(2024, 1, 1)],
        }
    )
    right = pl.DataFrame(
        {
            "id": [2],
            "company_name": ["right"],
            "postcode": ["SW1A 1AA"],
            "incorporation_date": [date(2024, 1, 1)],
            "duns_number": ["123456789"],
        }
    )

    fake_linker = Mock()
    with patch(
        "matchlab.models.linkers.splinklinker.SplinkLibLinkerClass",
        return_value=fake_linker,
    ) as splink_linker:
        linker.prepare(left, right)

    left_pd, right_pd = splink_linker.call_args.kwargs["input_table_or_tables"]
    assert list(left_pd.columns) == list(right_pd.columns)
    assert list(left_pd.columns) == [
        "id",
        "postcode",
        "duns_number",
        "company_name",
        "incorporation_date",
    ]
