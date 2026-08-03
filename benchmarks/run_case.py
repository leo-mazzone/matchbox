"""Measure one case, in a process of its own. Not meant to be run by hand.

`measure.run` spawns this with a case on stdin and reads a measurement off stdout. The
separate process is what makes peak memory mean anything — see `measure` for why — and
it also means one case cannot leave a warmed allocator, an open DuckDB connection or a
populated cache behind for the next one.

Everything here goes through matchlab's public API. The point of a benchmark is to
measure what a user can reach.
"""

import json
import resource
import sys
import tempfile
import time
from pathlib import Path

from benchmarks.cases import Case, Measurement, Store
from benchmarks.topologies import build, declare
from matchlab.adapters import DuckDBAdapter


def peak_rss() -> int:
    """This process's high-water resident set, in bytes.

    `getrusage` rather than `tracemalloc`: almost none of the memory here is Python's.
    It is DuckDB's buffer pool, Polars' frames and Arrow's buffers, and `tracemalloc`
    is blind to all three. This is a high-water mark rather than a reading, which is
    the right shape of number for "will this fit" and the wrong one for "what is it
    holding now".

    `ru_maxrss` is in **bytes on macOS and kilobytes on Linux** — a difference of three
    orders of magnitude that would otherwise pass silently, since both figures look
    plausible.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def measure(case: Case, path: Path, warehouse_rows: int) -> Measurement:
    """Run this case over an already-generated warehouse and record what it cost.

    Three phases, all public API. `collect_warm` rebuilds the plan from scratch before
    collecting it into the same store, because reusing the collected object would
    short-circuit on its own fingerprints and measure nothing — a fresh plan has to
    re-derive every fingerprint and ask the store, which is what a second run of a
    script actually does.

    Raises:
        AssertionError: If the resolution is not the size the shape guarantees. A fast
            wrong answer is worse than a slow one, and without this the suite could
            report a beautifully scaling pipeline that had stopped matching anything.
    """
    with tempfile.TemporaryDirectory() as directory:
        location = (
            ":memory:"
            if case.store is Store.MEMORY
            else str(Path(directory) / "store.duckdb")
        )
        store = DuckDBAdapter(location)
        try:
            plan = build(case.topology, declare(path, case.warehouse.sources))

            start = time.perf_counter()
            plan.collect(store, interactive=False)
            cold = time.perf_counter() - start

            start = time.perf_counter()
            entities = plan.entities()
            read = time.perf_counter() - start

            again = build(case.topology, declare(path, case.warehouse.sources))
            start = time.perf_counter()
            again.collect(store, interactive=False)
            warm = time.perf_counter() - start

            # Rows the plan actually read, not rows the warehouse holds: `DEDUPE`
            # reads one source of several, and charging it for the rest would make its
            # per-row column a quarter of the truth.
            n_rows = entities.height
            resolved = entities["root"].n_unique()
            expected = case.expected_entities
            assert expected is None or resolved == expected, (
                f"{case.topology} resolved {resolved:,} entities, expected "
                f"{expected:,}. The pipeline is not matching what it should, so its "
                "timings describe nothing."
            )
            assert resolved <= n_rows <= warehouse_rows, (
                f"{resolved:,} entities over {n_rows:,} rows of a {warehouse_rows:,} "
                "row warehouse is impossible."
            )

            return Measurement(
                n_rows=n_rows,
                n_steps=len(plan.lineage()),
                collect_cold_s=cold,
                entities_s=read,
                collect_warm_s=warm,
                peak_rss_bytes=peak_rss(),
                store_bytes=store.stats().bytes,
                n_entities_resolved=resolved,
            )
        finally:
            store.close()


def main() -> None:
    """Read a case from stdin, write a measurement to stdout.

    JSON both ways, and nothing else on stdout — the parent parses the whole stream,
    so a stray `print` here would look like a corrupt measurement. matchlab's own
    logging goes to stderr, which the parent surfaces only when a case fails.
    """
    request = json.loads(sys.stdin.read())

    if request.get("warmup"):
        # A case-shaped no-op, so the parent can price interpreter startup and the
        # matchlab import and keep them out of every case's figures.
        print(json.dumps({"peak_rss_bytes": peak_rss()}))
        return

    case = Case.model_validate(request["case"])
    measurement = measure(case, Path(request["path"]), request["warehouse_rows"])
    print(measurement.model_dump_json())


if __name__ == "__main__":
    main()
