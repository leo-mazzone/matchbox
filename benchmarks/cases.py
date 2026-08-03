"""What gets measured, and what a measurement is.

The sweeps here vary **one axis at a time** from a shared baseline rather than crossing
every axis with every other, to save time and to isolate the impact of each axis.
"""

from collections.abc import Sequence
from enum import StrEnum
from statistics import median

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.data import Warehouse
from benchmarks.topologies import Topology

#: The middle of every sweep. Each sweep replaces exactly one of these.
BASELINE = Warehouse(n_entities=50_000, records_per_entity=3, n_sources=4)
BASELINE_TOPOLOGY = Topology.MESH


class Store(StrEnum):
    """Where a case's artifacts go while it runs.

    `FILE` is what a user gets by default and makes store size a real number on a real
    disk. `MEMORY` keeps everything in the process, which moves the same bytes into
    peak RSS — the same work, attributed differently, which is the only reason to
    measure both.
    """

    FILE = "file"
    MEMORY = "memory"


class Case(BaseModel):
    """One plan shape over one warehouse: the unit of measurement.

    Frozen and hashable so the runner can group cases by warehouse and build each one
    only once, however many topologies read it.
    """

    model_config = ConfigDict(frozen=True)

    topology: Topology
    warehouse: Warehouse
    store: Store = Store.FILE

    @property
    def label(self) -> str:
        """This case in one line, for a progress record to name."""
        return (
            f"{self.topology} / {self.warehouse.n_entities:,} entities / "
            f"{self.warehouse.records_per_entity} per entity / "
            f"{self.warehouse.n_sources} sources"
        )

    @property
    def expected_entities(self) -> int | None:
        """How many entities this shape must resolve to, where the shape fixes it.

        Every entity's records clean to one value, so any shape that can reach every
        pair of sources recovers the planted entities exactly. `DEDUPE` sees one
        source, so it recovers only what that source holds.

        `HUB` is the exception and returns `None`: it cannot merge two spokes that
        share an entity the hub has never heard of, so how many entities it resolves
        depends on the overlap rather than on the shape. That is the property the
        topology sweep exists to show, so it is reported, never asserted.
        """
        match self.topology:
            case Topology.DEDUPE:
                return self.warehouse.covered
            case Topology.CHAIN | Topology.MESH:
                return self.warehouse.n_entities
            case Topology.HUB:
                return None


class Measurement(BaseModel):
    """What one run of one case cost.

    Attributes:
        n_rows: Rows in the warehouse it read.
        n_steps: Steps in the plan, so a shape's cost can be read per step.
        collect_cold_s: Materialising the whole plan into an empty store.
        entities_s: Reading the complete resolution back out of it.
        collect_warm_s: Collecting the same plan again into the same store. Every step
            is a cache hit, so this is the price of establishing that nothing changed —
            the number that matters if you iterate, which matching is.
        peak_rss_bytes: The measuring process's high-water resident set, over
            everything above. Includes the store when it is in memory.
        store_bytes: What the store weighs afterwards.
        n_entities_resolved: How many entities came out. Reported for every case,
            asserted only where `Case.expected_entities` fixes it.
    """

    model_config = ConfigDict(frozen=True)

    n_rows: int
    n_steps: int
    collect_cold_s: float
    entities_s: float
    collect_warm_s: float
    peak_rss_bytes: int
    store_bytes: int
    n_entities_resolved: int


class Result(BaseModel):
    """A case and every repeat of it.

    Repeats are kept rather than averaged away: a run's spread is the only evidence
    that its middle means anything, and a JSON file that threw it out could not be
    used to argue about a regression later.
    """

    model_config = ConfigDict(frozen=True)

    case: Case
    runs: tuple[Measurement, ...] = Field(min_length=1)

    def middle(self, field: str) -> float:
        """The median of `field` across repeats.

        Median rather than mean, because the failure mode on a laptop is one run being
        interrupted by something else entirely, and a mean carries that forever.
        """
        return median(getattr(run, field) for run in self.runs)

    @property
    def rows(self) -> int:
        """Rows read. Fixed by the warehouse, so any run answers."""
        return self.runs[0].n_rows

    @property
    def steps(self) -> int:
        """Steps in the plan. Fixed by the topology, so any run answers."""
        return self.runs[0].n_steps

    @property
    def us_per_row(self) -> float:
        """Microseconds of cold collect per row read.

        The column that makes three orders of magnitude comparable: flat means linear
        scaling, and rising means something is quadratic in the data.
        """
        return self.middle("collect_cold_s") * 1_000_000 / self.rows

    def bytes_per_row(self, baseline_rss: int) -> float:
        """Resident bytes per row, above what an idle process already holds.

        The baseline has to come out, and it is not a rounding correction: a fresh
        interpreter with matchlab imported is a few hundred megabytes before it reads
        anything, which at small sizes is the entire figure. What is left is the part
        that scales with the data, and flat is what linear memory looks like.
        """
        return max(self.middle("peak_rss_bytes") - baseline_rss, 0) / self.rows

    def axis(self, column: str) -> str:
        """This case's value on the axis a sweep varies, ready to print."""
        match column:
            case "topology":
                return str(self.case.topology)
            case "entities":
                return f"{self.case.warehouse.n_entities:,}"
            case "records":
                return str(self.case.warehouse.records_per_entity)
            case "sources":
                return f"{self.case.warehouse.n_sources} / {self.case.topology}"
            case _:
                raise ValueError(f"No axis called '{column}'.")


class Sweep(BaseModel):
    """One question, and the cases that answer it.

    Attributes:
        title: The axis being varied.
        holding: What is held fixed, so a reader can see the table is honest.
        column: Which per-case value the first table column shows.
        cases: The cases, in the order they should be read.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    holding: str
    column: str
    cases: tuple[Case, ...] = Field(min_length=1)


class Environment(BaseModel):
    """The machine and the build a run happened on.

    Recorded because a benchmark result without it is not comparable with anything:
    two JSON files can only be diffed once you can see whether they came off the same
    hardware and the same commit.
    """

    model_config = ConfigDict(frozen=True)

    started_at: str
    profile: str
    repeats: int
    store: Store
    matchlab_version: str
    git_sha: str
    python: str
    platform: str
    cpus: int
    startup_s: float
    baseline_rss_bytes: int


class SweepResult(BaseModel):
    """One sweep, measured. Carries the question so the table can restate it."""

    model_config = ConfigDict(frozen=True)

    title: str
    holding: str
    column: str
    results: tuple[Result, ...] = Field(min_length=1)


class Run(BaseModel):
    """A whole run: what the machine was, and what every case cost.

    This is exactly what gets written to `results/`, so the file a later run is diffed
    against holds every raw repeat rather than the medians that were printed.
    """

    model_config = ConfigDict(frozen=True)

    environment: Environment
    sweeps: tuple[SweepResult, ...]

    def find(self, title: str) -> SweepResult | None:
        """The sweep with this title, if this run included it."""
        return next((sweep for sweep in self.sweeps if sweep.title == title), None)


def _fixed(warehouse: Warehouse, topology: Topology, *, but: str) -> str:
    """Describe everything about a case except the axis being swept."""
    parts = {
        "topology": str(topology),
        "entities": f"{warehouse.n_entities:,} entities",
        "records": f"{warehouse.records_per_entity} records/entity",
        "sources": f"{warehouse.n_sources} sources",
    }
    del parts[but]
    return ", ".join(parts.values())


def topology_sweep(warehouse: Warehouse, store: Store) -> Sweep:
    """Every shape over one warehouse. The headline table."""
    return Sweep(
        title="Plan topology",
        holding=_fixed(warehouse, BASELINE_TOPOLOGY, but="topology"),
        column="topology",
        cases=tuple(
            Case(topology=topology, warehouse=warehouse, store=store)
            for topology in Topology
        ),
    )


def entity_sweep(counts: Sequence[int], store: Store) -> Sweep:
    """More entities, same records each. Grows the resolution, not the duplication."""
    warehouses = [BASELINE.model_copy(update={"n_entities": count}) for count in counts]
    return Sweep(
        title="Number of entities",
        holding=_fixed(warehouses[0], BASELINE_TOPOLOGY, but="entities"),
        column="entities",
        cases=tuple(
            Case(topology=BASELINE_TOPOLOGY, warehouse=warehouse, store=store)
            for warehouse in warehouses
        ),
    )


def record_sweep(counts: Sequence[int], store: Store) -> Sweep:
    """More records per entity, same entities. Grows duplication, not the answer."""
    warehouses = [
        BASELINE.model_copy(update={"records_per_entity": count}) for count in counts
    ]
    return Sweep(
        title="Records per entity",
        holding=_fixed(warehouses[0], BASELINE_TOPOLOGY, but="records"),
        column="records",
        cases=tuple(
            Case(topology=BASELINE_TOPOLOGY, warehouse=warehouse, store=store)
            for warehouse in warehouses
        ),
    )


def source_sweep(counts: Sequence[int], store: Store) -> Sweep:
    """More sources, for both shapes whose link count depends on how many there are.

    The pair is the point: a mesh links every pair and a hub links `n-1`, so this is
    where the quadratic and the linear shape visibly part company.
    """
    cases = tuple(
        Case(
            topology=topology,
            warehouse=BASELINE.model_copy(update={"n_sources": count}),
            store=store,
        )
        for count in counts
        for topology in (Topology.HUB, Topology.MESH)
    )
    return Sweep(
        title="Number of sources",
        holding=f"{BASELINE.n_entities:,} entities, "
        f"{BASELINE.records_per_entity} records/entity, hub and mesh",
        column="sources",
        cases=cases,
    )


def profile(name: str, store: Store = Store.FILE) -> list[Sweep]:
    """The sweeps a named profile runs.

    `quick` is a smoke test — every shape, small enough to finish in under a minute,
    and the one to run before believing anything else here. `standard` is the run
    worth quoting. `full` pushes each axis to where the curves either stay straight or
    stop being straight, and takes long enough that you should start it and go away.

    Raises:
        KeyError: If the profile is not one of the three.
    """
    small = BASELINE.model_copy(update={"n_entities": 5_000, "records_per_entity": 2})

    profiles: dict[str, list[Sweep]] = {
        "quick": [topology_sweep(small, store)],
        "standard": [
            topology_sweep(BASELINE, store),
            entity_sweep([10_000, 50_000, 200_000], store),
            record_sweep([1, 3, 6], store),
            source_sweep([2, 4, 6], store),
        ],
        "full": [
            topology_sweep(BASELINE, store),
            entity_sweep([10_000, 50_000, 200_000, 1_000_000], store),
            record_sweep([1, 3, 6, 12], store),
            source_sweep([2, 4, 6, 8], store),
        ],
    }
    if name not in profiles:
        raise KeyError(f"Unknown profile '{name}'. Choose from {sorted(profiles)}.")
    return profiles[name]


PROFILES = ("quick", "standard", "full")
