"""Drive the cases, one subprocess each, and collect what they cost.

Isolation is the whole design here, and it buys two things.

**Peak memory becomes attributable.** A high-water resident set is monotonic within a
process, so a second case measured in the first one's process inherits its peak and
reports a number that belongs to neither. One process per case means each reports only
its own.

**Cases stop contaminating each other.** A warmed allocator, a DuckDB connection left
open, a Polars thread pool already spun up, an lru_cache holding a frame — all of them
make whichever case runs second look faster than it is.

What isolation costs is interpreter startup, paid once per case. It is priced at the
start of a run and reported in the header rather than folded into any case, so nobody
has to wonder whether a small case's number is mostly Python starting up.

The warehouse is generated **here**, in the parent, and only its path is passed down.
Generating inside a measured child would put the generator's allocations into that
child's peak, and the generator is not the thing under measurement.
"""

import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence

from benchmarks.cases import Case, Measurement, Result, Sweep
from benchmarks.data import Warehouse, ensure

#: How long one case may take before it is assumed wedged rather than slow. Generous:
#: the largest case in `full` is minutes of real work.
DEFAULT_TIMEOUT = 3600


class CaseFailed(RuntimeError):
    """A case did not produce a measurement.

    Carries the child's stderr, because that is where the traceback is — including the
    assertion that fires when a plan stops resolving what it should, which is the one
    failure worth reading in full.
    """

    def __init__(self, label: str, detail: str) -> None:
        """Record which case failed, and everything the child said about why."""
        super().__init__(f"{label} failed:\n{detail}")
        self.label = label
        self.detail = detail


def _child(request: dict, label: str, timeout: int) -> tuple[dict, float]:
    """Run one child to completion and return what it said, and how long it took.

    Raises:
        CaseFailed: If the child exits non-zero, times out, or says something that is
            not a measurement.
    """
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "benchmarks.run_case"],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise CaseFailed(label, f"timed out after {timeout}s") from error
    except subprocess.CalledProcessError as error:
        raise CaseFailed(label, error.stderr or "no output") from error

    elapsed = time.perf_counter() - start
    try:
        return json.loads(completed.stdout), elapsed
    except json.JSONDecodeError as error:
        raise CaseFailed(label, f"unparseable output: {completed.stdout!r}") from error


def baseline() -> tuple[float, int]:
    """What a child costs before it has done anything: seconds, and resident bytes.

    Priced by running a child that does nothing but import matchlab and answer. Both
    figures are floors under every case, and both are large enough to mislead if they
    are not named: the interpreter and its extension modules are a few hundred
    megabytes resident before a single row is read, so a peak RSS quoted without this
    would say a small case used far more memory than it did.

    Reported in the header, and subtracted to give the per-row memory column.
    """
    payload, elapsed = _child({"warmup": True}, label="startup", timeout=300)
    return elapsed, int(payload["peak_rss_bytes"])


def warehouses(sweeps: Iterable[Sweep]) -> list[Warehouse]:
    """Every distinct warehouse a set of sweeps needs, in the order first wanted.

    Distinct, because a topology sweep runs four shapes over one warehouse and the
    source sweep runs two — generating per case would be most of the run.
    """
    seen: dict[Warehouse, None] = {}
    for sweep in sweeps:
        for case in sweep.cases:
            seen.setdefault(case.warehouse)
    return list(seen)


def prepare(
    sweeps: Sequence[Sweep], on_start: Callable[[Warehouse], None] | None = None
) -> dict[Warehouse, tuple[str, int]]:
    """Generate every warehouse the sweeps need, and report where each landed.

    Cached between runs, so only a shape never generated before costs anything.

    Returns:
        Warehouse to its path and its actual row count.
    """
    built: dict[Warehouse, tuple[str, int]] = {}
    for warehouse in warehouses(sweeps):
        if on_start is not None:
            on_start(warehouse)
        path, rows = ensure(warehouse)
        built[warehouse] = (str(path), rows)
    return built


def run(
    case: Case,
    path: str,
    warehouse_rows: int,
    repeats: int = 1,
    timeout: int = DEFAULT_TIMEOUT,
    on_repeat: Callable[[int, Measurement], None] | None = None,
) -> Result:
    """Measure one case, `repeats` times, each in a fresh process.

    Args:
        case: What to measure.
        path: The warehouse it reads, already generated.
        warehouse_rows: How many rows that warehouse holds. An upper bound the child
            checks itself against; what it reports is the rows its plan actually read.
        repeats: How many times to run it. The median is reported; every run is kept.
        timeout: Seconds one repeat may take.
        on_repeat: Called after each repeat, so a caller can show progress.

    Raises:
        CaseFailed: If any repeat fails. A partial result would be a lie about the
            middle, so one bad repeat fails the case.
    """
    runs: list[Measurement] = []
    for attempt in range(repeats):
        payload, _ = _child(
            {
                "case": case.model_dump(mode="json"),
                "path": path,
                "warehouse_rows": warehouse_rows,
            },
            label=case.label,
            timeout=timeout,
        )
        measurement = Measurement.model_validate(payload)
        runs.append(measurement)
        if on_repeat is not None:
            on_repeat(attempt, measurement)

    return Result(case=case, runs=tuple(runs))
