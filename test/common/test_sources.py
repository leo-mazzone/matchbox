import copy

import pandas as pd
import pyarrow as pa
import pytest
from pandas.testing import assert_frame_equal
from sqlalchemy import Engine, Table, create_engine

from matchbox.client.helpers.selector import Match
from matchbox.common.db import fullname_to_prefix
from matchbox.common.exceptions import MatchboxSourceColumnError
from matchbox.common.factories.sources import source_factory
from matchbox.common.sources import Source, SourceAddress, SourceColumn


def test_source_address_compose():
    """Correct addresses are generated from engines and table names"""
    pg = create_engine("postgresql://user:fakepass@host:1234/db")  # trufflehog:ignore
    pg_host = create_engine(
        "postgresql://user:fakepass@host2:1234/db"  # trufflehog:ignore
    )
    pg_port = create_engine(
        "postgresql://user:fakepass@host:4321/db"  # trufflehog:ignore
    )
    pg_db = create_engine(
        "postgresql://user:fakepass@host:1234/db2"  # trufflehog:ignore
    )
    pg_user = create_engine(
        "postgresql://user2:fakepass@host:1234/db"  # trufflehog:ignore
    )
    pg_password = create_engine(
        "postgresql://user:fakepass2@host:1234/db"  # trufflehog:ignore
    )
    pg_dialect = create_engine(
        "postgresql+psycopg2://user:fakepass@host:1234/db"  # trufflehog:ignore
    )

    sqlite = create_engine("sqlite:///foo.db")
    sqlite_name = create_engine("sqlite:///bar.db")

    different_wh_hashes = set(
        [
            SourceAddress.compose(pg, "tablename").warehouse_hash,
            SourceAddress.compose(pg_host, "tablename").warehouse_hash,
            SourceAddress.compose(pg_port, "tablename").warehouse_hash,
            SourceAddress.compose(pg_db, "tablename").warehouse_hash,
            SourceAddress.compose(sqlite, "tablename").warehouse_hash,
            SourceAddress.compose(sqlite_name, "tablename").warehouse_hash,
        ]
    )

    assert len(different_wh_hashes) == 6

    same_wh_hashes = set(
        [
            SourceAddress.compose(pg, "tablename").warehouse_hash,
            SourceAddress.compose(pg_user, "tablename").warehouse_hash,
            SourceAddress.compose(pg_password, "tablename").warehouse_hash,
            SourceAddress.compose(pg_dialect, "tablename").warehouse_hash,
        ]
    )

    assert len(same_wh_hashes) == 1

    same_table_name = set(
        [
            SourceAddress.compose(pg, "tablename").full_name,
            SourceAddress.compose(sqlite, "tablename").full_name,
        ]
    )

    assert len(same_table_name) == 1


def test_source_set_engine(sqlite_warehouse: Engine):
    """Engine can be set on Source"""
    source_testkit = source_factory(
        features=[{"name": "b", "base_generator": "random_int", "sql_type": "BIGINT"}],
        engine=sqlite_warehouse,
    )
    source_testkit.to_warehouse(engine=sqlite_warehouse)

    # We can set engine with correct column specification
    source = source_testkit.source.set_engine(sqlite_warehouse)
    assert isinstance(source, Source)

    # Error is raised with missing column
    with pytest.raises(MatchboxSourceColumnError, match="Column c not available in"):
        new_source = copy.copy(source_testkit.source)
        new_source.columns = [SourceColumn(name="c", type="TEXT")]
        new_source.set_engine(sqlite_warehouse)

    # Error is raised with wrong type
    with pytest.raises(MatchboxSourceColumnError, match="Type BIGINT != TEXT for b"):
        new_source = copy.copy(source_testkit.source)
        new_source.columns = [SourceColumn(name="b", type="TEXT")]
        new_source.set_engine(sqlite_warehouse)


def test_source_signature():
    """Source signatures are generated correctly"""
    # Column order doesn't matter
    source1 = Source(
        address=SourceAddress(full_name="foo", warehouse_hash=b"bar1"),
        db_pk="i",
        columns=[
            SourceColumn(name="a", type="TEXT"),
            SourceColumn(name="b", type="TEXT"),
        ],
    )
    source2 = Source(
        address=SourceAddress(full_name="foo", warehouse_hash=b"bar2"),
        db_pk="i",
        columns=[
            SourceColumn(name="b", type="TEXT"),
            SourceColumn(name="a", type="TEXT"),
        ],
    )
    assert source1.signature == source2.signature

    # Column type matters
    source1 = Source(
        address=SourceAddress(full_name="foo", warehouse_hash=b"bar1"),
        db_pk="i",
        columns=[
            SourceColumn(name="a", type="TEXT"),
        ],
    )
    source2 = Source(
        address=SourceAddress(full_name="foo", warehouse_hash=b"bar2"),
        db_pk="i",
        columns=[
            SourceColumn(name="a", type="BIGINT"),
        ],
    )
    assert source1.signature != source2.signature

    # Table name matters
    source1 = Source(
        address=SourceAddress(full_name="bar", warehouse_hash=b"bar1"),
        db_pk="i",
        columns=[
            SourceColumn(name="a", type="TEXT"),
        ],
    )
    source2 = Source(
        address=SourceAddress(full_name="foo", warehouse_hash=b"bar2"),
        db_pk="i",
        columns=[
            SourceColumn(name="a", type="TEXT"),
        ],
    )
    assert source1.signature != source2.signature

    # Alias supersedes table name
    source1 = Source(
        alias="alias",
        address=SourceAddress(full_name="bar", warehouse_hash=b"bar1"),
        db_pk="i",
        columns=[
            SourceColumn(name="a", type="TEXT"),
        ],
    )
    source2 = Source(
        alias="alias",
        address=SourceAddress(full_name="foo", warehouse_hash=b"bar2"),
        db_pk="i",
        columns=[
            SourceColumn(name="a", type="TEXT"),
        ],
    )
    assert source1.signature == source2.signature

    # Column name matters
    source1 = Source(
        address=SourceAddress(full_name="foo", warehouse_hash=b"bar1"),
        db_pk="i",
        columns=[
            SourceColumn(name="a", type="TEXT"),
        ],
    )
    source2 = Source(
        address=SourceAddress(full_name="foo", warehouse_hash=b"bar2"),
        db_pk="i",
        columns=[
            SourceColumn(name="b", type="TEXT"),
        ],
    )
    assert source1.signature != source2.signature

    # Alias supersedes column name
    source1 = Source(
        address=SourceAddress(full_name="foo", warehouse_hash=b"bar1"),
        db_pk="i",
        columns=[
            SourceColumn(name="a", alias="alias", type="TEXT"),
        ],
    )
    source2 = Source(
        address=SourceAddress(full_name="foo", warehouse_hash=b"bar2"),
        db_pk="i",
        columns=[
            SourceColumn(name="b", alias="alias", type="TEXT"),
        ],
    )
    assert source1.signature == source2.signature


def test_source_format_columns():
    """Column names can get a standard prefix from a table name"""
    source1 = Source(
        address=SourceAddress(full_name="foo", warehouse_hash=b"bar"), db_pk="i"
    )

    source2 = Source(
        address=SourceAddress(full_name="foo.bar", warehouse_hash=b"bar"), db_pk="i"
    )

    assert source1.format_column("col") == "foo_col"
    assert source2.format_column("col") == "foo_bar_col"


def test_source_default_columns(sqlite_warehouse: Engine):
    """Default columns from the warehouse can be assigned to a Source."""
    source_testkit = source_factory(
        features=[
            {"name": "a", "base_generator": "random_int", "sql_type": "BIGINT"},
            {"name": "b", "base_generator": "word", "sql_type": "TEXT"},
        ],
        engine=sqlite_warehouse,
    )

    source_testkit.to_warehouse(engine=sqlite_warehouse)

    expected_columns = [
        SourceColumn(name="a", type="BIGINT"),
        SourceColumn(name="b", type="TEXT"),
    ]

    source = source_testkit.source.set_engine(sqlite_warehouse).default_columns()

    assert source.columns == expected_columns


def test_source_to_table(sqlite_warehouse: Engine):
    """Convert Source to SQLAlchemy Table."""
    source_testkit = source_factory(engine=sqlite_warehouse)
    source_testkit.to_warehouse(engine=sqlite_warehouse)

    source = source_testkit.source.set_engine(sqlite_warehouse)

    assert isinstance(source.to_table(), Table)


def test_source_to_arrow_to_pandas(sqlite_warehouse: Engine):
    """Convert Source to Arrow table or Pandas dataframe with options."""
    source_testkit = source_factory(
        features=[
            {"name": "a", "base_generator": "random_int", "sql_type": "BIGINT"},
            {"name": "b", "base_generator": "word", "sql_type": "TEXT"},
        ],
        engine=sqlite_warehouse,
        n_true_entities=2,
    )
    source_testkit.to_warehouse(engine=sqlite_warehouse)
    source = source_testkit.source.set_engine(sqlite_warehouse).default_columns()
    prefix = fullname_to_prefix(source_testkit.name)
    expected_df_prefixed = (
        source_testkit.data.to_pandas().drop(columns=["id"]).add_prefix(prefix)
    )

    # Test basic conversion
    assert_frame_equal(
        expected_df_prefixed, source.to_pandas(), check_like=True, check_dtype=False
    )
    assert_frame_equal(
        expected_df_prefixed,
        source.to_arrow().to_pandas(),
        check_like=True,
        check_dtype=False,
    )

    # Test with limit parameter
    assert_frame_equal(
        expected_df_prefixed.iloc[:1],
        source.to_pandas(limit=1),
        check_like=True,
        check_dtype=False,
    )
    assert_frame_equal(
        expected_df_prefixed.iloc[:1],
        source.to_arrow(limit=1).to_pandas(),
        check_like=True,
        check_dtype=False,
    )

    # Test with fields parameter
    assert_frame_equal(
        expected_df_prefixed[[f"{prefix}pk", f"{prefix}a"]],
        source.to_pandas(fields=["a"]),
        check_like=True,
        check_dtype=False,
    )
    assert_frame_equal(
        expected_df_prefixed[[f"{prefix}pk", f"{prefix}a"]],
        source.to_arrow(fields=["a"]).to_pandas(),
        check_like=True,
        check_dtype=False,
    )


def test_source_hash_data(sqlite_warehouse: Engine):
    """A Source can output hashed versions of its rows."""
    source_testkit = source_factory(
        features=[
            {"name": "a", "base_generator": "random_int", "sql_type": "BIGINT"},
            {"name": "b", "base_generator": "word", "sql_type": "TEXT"},
        ],
        engine=sqlite_warehouse,
        n_true_entities=2,
        repetition=1,
    )

    source_testkit.to_warehouse(engine=sqlite_warehouse)
    source = source_testkit.source.set_engine(sqlite_warehouse).default_columns()

    res = source.hash_data().to_pandas()
    assert len(res) == 2
    assert len(res.source_pk.iloc[0]) == 2
    assert len(res.source_pk.iloc[1]) == 2

    result = source.hash_data(iter_batches=True, batch_size=3)
    assert isinstance(result, pa.Table)


@pytest.mark.parametrize(
    ("method_name", "return_type"),
    [
        pytest.param("to_arrow", pa.Table, id="to_arrow"),
        pytest.param("to_pandas", pd.DataFrame, id="to_pandas"),
    ],
)
def test_source_data_batching(method_name, return_type, sqlite_warehouse: Engine):
    """Test Source data retrieval methods with batching parameters."""
    # Create a source with multiple rows of data
    source_testkit = source_factory(
        features=[
            {"name": "a", "base_generator": "random_int", "sql_type": "BIGINT"},
            {"name": "b", "base_generator": "word", "sql_type": "TEXT"},
        ],
        engine=sqlite_warehouse,
        n_true_entities=9,
    )
    source_testkit.to_warehouse(engine=sqlite_warehouse)
    source = source_testkit.source.set_engine(sqlite_warehouse).default_columns()

    # Call the method with batching
    method = getattr(source, method_name)
    batch_iterator = method(iter_batches=True, batch_size=3)
    batches = list(batch_iterator)

    # Verify we got the expected number of batches
    assert len(batches) == 3
    for batch in batches:
        assert isinstance(batch, return_type)
        assert len(batch) == 3


def test_match_validates():
    """Match objects are validated when they're instantiated."""
    Match(
        cluster=1,
        source=SourceAddress(full_name="test.source", warehouse_hash=b"bar"),
        source_id={"a"},
        target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
        target_id={"b"},
    )

    # Missing source_id with target_id
    with pytest.raises(ValueError):
        Match(
            cluster=1,
            source=SourceAddress(full_name="test.source", warehouse_hash=b"bar"),
            target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
            target_id={"b"},
        )

    # Missing cluster with target_id
    with pytest.raises(ValueError):
        Match(
            source=SourceAddress(full_name="test.source", warehouse_hash=b"bar"),
            source_id={"a"},
            target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
            target_id={"b"},
        )

    # Missing source_id with cluster
    with pytest.raises(ValueError):
        Match(
            cluster=1,
            source=SourceAddress(full_name="test.source", warehouse_hash=b"bar"),
            target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
        )
