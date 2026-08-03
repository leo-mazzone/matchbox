"""Generate a warehouse of a known size and shape, and cache it on disk.

The two things a benchmark wants to vary independently are how many *entities* exist
and how many *records* describe each one. Those cost different things — entities drive
the size of every resolution artifact, records drive how much the matchers chew
through — so conflating them into a single `n_rows` knob, as the old example benchmark
did, makes the resulting curve uninterpretable. Here they are separate arguments.

Generation is pure Polars expressions over an entity index: no Python loop, no RNG
object, no Faker. `matchlab.testkit` is the right tool when you need planted
`TrueEntity`s and a scored diff, and the wrong one here — it generates entity by entity
through Faker, which does not reach a million entities in a time anyone will wait for.
What the benchmark needs from ground truth is much weaker: a `_truth` column, so a case
can tell whether it resolved something sane before reporting how fast it did it.

Warehouses are cached under the user cache directory, keyed by the parameters that
produced them. One warehouse serves every topology at a given size, so a sweep pays
generation once, and it is built by the parent process — a child measuring peak RSS
should not be carrying the generator's allocations in its high-water mark.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import polars as pl
from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import create_engine

#: Cosmetic suffixes the cleaning rules are meant to absorb — all of them clean away
#: to nothing, so a name carrying any of them compares equal to a name carrying none.
SUFFIXES = ("LTD", "LIMITED", "PLC", "", "Ltd.", "ltd")

#: Runs of whitespace, which cleaning collapses. Sloppy spacing is the most common
#: thing that makes two renderings of one name differ without meaning anything.
SPACES = (" ", "  ", "   ")

#: Towns, for postcodes that look like postcodes. Only the two-letter prefix is used.
TOWNS = ("LONDON", "MANCHESTER", "LEEDS", "BRISTOL", "EDINBURGH", "GLASGOW")

# How a rendering is decomposed: the variation number is read as a mixed-radix
# integer, one digit per cosmetic choice, so two variations agree on every choice only
# once the whole space is exhausted. Each constant is the place value of its digit.
_NAME_CASE = 1
_SUFFIX = _NAME_CASE * 2
_NAME_SPACE = _SUFFIX * len(SUFFIXES)
_POSTCODE_SPACE = _NAME_SPACE * len(SPACES)
_POSTCODE_CASE = _POSTCODE_SPACE * len(SPACES)

#: How many distinct renderings of one entity exist. Every row of an entity gets its
#: own, so `n_sources * records_per_entity` may not exceed this.
VARIATIONS = _POSTCODE_CASE * 2

#: Bump this whenever the generated *values* change for unchanged parameters.
#: Warehouses are cached on disk by their parameters, and parameters alone cannot tell
#: a cached file that the code which wrote it has since changed — the run would
#: silently measure last month's data. Folding a version into the key retires the old
#: file instead.
GENERATION = 2

#: Source names, in the order topologies consume them. The first is the hub.
SOURCE_NAMES = ("crn", "dh", "cdms", "hmrc", "vat", "eori", "sic", "fca")

#: Knuth's multiplicative constant. Deterministic pseudo-randomness that vectorises:
#: reproducible across runs and platforms, and free compared with seeding a real RNG
#: per row.
KNUTH = 2654435761


class Warehouse(BaseModel):
    """The parameters that fully determine a generated warehouse, and where it landed.

    Frozen and hashable, because the sweep uses it as a cache key: two cases wanting
    the same data must agree on one file rather than each building their own.

    Attributes:
        n_entities: How many real-world things exist. Every source draws from these.
        records_per_entity: How many rows each source holds for an entity it knows
            about. The duplicates within a source are what deduplication removes.
        n_sources: How many sources to build. Capped by `SOURCE_NAMES`.
        coverage: The fraction of entities each source knows about, as a proportion.
            Below 1.0 the sources only partly overlap, which is what makes linking
            them worth doing — and what makes plan topology matter, since a shape that
            cannot reach two sources directly resolves fewer entities.
        seed: Shifts the pseudo-random stream, so a run can be repeated on different
            data of the same shape.
    """

    model_config = ConfigDict(frozen=True)

    n_entities: int = Field(gt=0)
    records_per_entity: int = Field(gt=0)
    n_sources: int = Field(gt=1, le=len(SOURCE_NAMES))
    coverage: float = Field(default=0.65, gt=0.0, le=1.0)
    seed: int = 0

    @model_validator(mode="after")
    def _renderings_are_distinct(self) -> Self:
        """Every row of an entity needs its own rendering, and they are finite.

        Beyond `VARIATIONS` rows per entity two of them would render identically,
        share a leaf cluster, and be merged before any model ran — quietly making the
        matching cheaper than the case claims to be measuring. Better to refuse.
        """
        rows_per_entity = self.n_sources * self.records_per_entity
        if rows_per_entity > VARIATIONS:
            raise ValueError(
                f"{self.n_sources} sources x {self.records_per_entity} records is "
                f"{rows_per_entity} rows per entity, and only {VARIATIONS} distinct "
                "renderings exist. Lower one of them, or add to SUFFIXES/SPACES."
            )
        return self

    @property
    def sources(self) -> tuple[str, ...]:
        """The source names this warehouse holds, hub first."""
        return SOURCE_NAMES[: self.n_sources]

    @property
    def covered(self) -> int:
        """Roughly how many entities any one source holds.

        Approximate by a handful either way: membership is decided by a modular
        filter over a pseudo-random draw, which lands near the coverage fraction
        rather than exactly on it.
        """
        return int(self.n_entities * self.coverage)

    @property
    def expected_rows(self) -> int:
        """Roughly how many rows this warehouse holds, before generating it.

        Approximate for the same reason `covered` is, and used only to say what is
        about to be generated. `ensure` returns the figure actually written, which is
        the one every measurement quotes.
        """
        return self.covered * self.records_per_entity * self.n_sources

    @property
    def path(self) -> Path:
        """Where this warehouse lives, keyed by what generated it.

        Content-keyed rather than named, so changing any parameter builds a new file
        instead of silently reusing data that no longer matches the case.
        """
        key = f"{GENERATION}:{self.model_dump_json(exclude_none=True)}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return user_cache_path("matchlab") / "benchmarks" / f"warehouse-{digest}.sqlite"


def generate(warehouse: Warehouse) -> dict[str, pl.DataFrame]:
    """Build one frame per source, in memory.

    Each source keeps a rotating slice of the entities, so no two sources hold quite
    the same set and the overlap between any pair is partial.

    Every row of an entity — across all its copies and all the sources that hold it —
    is rendered **differently**, and every rendering cleans to the same value. That is
    the point of `VARIATIONS`, and it is not decoration. matchlab content-addresses
    records: two rows with identical field values share a leaf cluster before any model
    runs. Render an entity the same way in two sources and they are already merged, the
    linkers have nothing to do, and every topology returns the same answer at the same
    price — which is exactly the measurement this suite exists to avoid taking.

    Returns:
        Source name to its rows: `key`, `name`, `postcode`, `_truth`.
    """
    base = pl.DataFrame({"entity": pl.arange(0, warehouse.n_entities, eager=True)})
    base = base.with_columns(
        pl.format("Company {}", pl.col("entity")).alias("stem"),
        (pl.col("entity") % len(TOWNS)).alias("town"),
        ((pl.col("entity") * KNUTH + warehouse.seed) % 1000).alias("draw"),
    )

    threshold = int(warehouse.coverage * 1000)
    span = 1000 // warehouse.n_sources

    frames: dict[str, pl.DataFrame] = {}
    for index, source in enumerate(warehouse.sources):
        # Rotating the draw per source is what makes coverage overlap rather than
        # nest: every source keeps the same *proportion* of entities, but a different
        # slice of them.
        present = base.filter((pl.col("draw") + index * span) % 1000 < threshold)

        rows = present.select(
            pl.col("entity"),
            pl.col("stem"),
            pl.col("town"),
            pl.int_ranges(0, warehouse.records_per_entity).alias("copy"),
        ).explode("copy", empty_as_null=False)

        # Unique per (source, copy), so no two rows of an entity render alike.
        variation = pl.col("copy") + index * warehouse.records_per_entity

        frames[source] = rows.select(
            pl.format(
                "{}-{}-{}", pl.lit(source), pl.col("entity"), pl.col("copy")
            ).alias("key"),
            _name(variation),
            _postcode(variation),
            pl.col("entity").alias("_truth"),
        )

    return frames


def _pick(options: Sequence[str], variation: pl.Expr, place: int) -> pl.Expr:
    """Choose one of `options` by the digit of `variation` at `place`.

    Indexing a literal series rather than `map_elements`, because a Python callback per
    row is the one thing that would make generation slower than the thing being
    measured.
    """
    return pl.lit(pl.Series(options)).gather(variation // place % len(options))


def _name(variation: pl.Expr) -> pl.Expr:
    """A company name that cleans to the same value however it is rendered."""
    cased = (
        pl.when(variation // _NAME_CASE % 2 == 0)
        .then(pl.col("stem").str.to_uppercase())
        .otherwise(pl.col("stem"))
    )
    return pl.format(
        "{}{}{}",
        cased,
        _pick(SPACES, variation, _NAME_SPACE),
        _pick(SUFFIXES, variation, _SUFFIX),
    ).alias("name")


def _postcode(variation: pl.Expr) -> pl.Expr:
    """A postcode, stable per entity, spaced and cased differently per rendering."""
    prefix = pl.lit(pl.Series([town[:2] for town in TOWNS])).gather(pl.col("town"))
    spaced = pl.format(
        "{}{}{}{}AA",
        prefix,
        pl.col("entity") % 90 + 1,
        _pick(SPACES, variation, _POSTCODE_SPACE),
        pl.col("entity") % 9 + 1,
    )
    return (
        pl.when(variation // _POSTCODE_CASE % 2 == 0)
        .then(spaced.str.to_uppercase())
        .otherwise(spaced.str.to_lowercase())
    ).alias("postcode")


def ensure(warehouse: Warehouse, rebuild: bool = False) -> tuple[Path, int]:
    """Return the path to this warehouse, building it if it is not already cached.

    Args:
        warehouse: What to generate.
        rebuild: Generate even if a cached file is already there.

    Returns:
        The database path, and how many rows it holds.
    """
    path = warehouse.path
    counted = path.with_suffix(".rows")

    if path.exists() and counted.exists() and not rebuild:
        return path, int(counted.read_text())

    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)

    engine = create_engine(f"sqlite:///{path}")
    total = 0
    try:
        for source, frame in generate(warehouse).items():
            frame.write_database(
                table_name=source, connection=engine, if_table_exists="replace"
            )
            total += frame.height
    finally:
        engine.dispose()

    # Alongside rather than inside, so counting rows never costs a query. The pair is
    # written last, so a crash mid-build leaves no file that looks complete.
    counted.write_text(str(total))
    return path, total
