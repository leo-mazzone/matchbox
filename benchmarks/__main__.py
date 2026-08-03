"""Run the benchmark suite: `just bench`, or `python -m benchmarks`.

    just bench                          # the standard profile
    just bench --profile quick          # a smoke test, under a minute
    just bench --profile full           # every axis pushed, go and have lunch
    just bench --only mesh --repeats 3  # one shape, three times

Generation happens first and up front. Cases are then measured one at a
time, each in its own process, printing as they finish.
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from benchmarks import measure, report
from benchmarks.cases import (
    PROFILES,
    Environment,
    Run,
    Store,
    SweepResult,
    profile,
)
from benchmarks.data import Warehouse
from benchmarks.measure import CaseFailed
from benchmarks.topologies import Topology
from matchlab.adapters.base import format_bytes

RESULTS = "benchmarks/results"


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(
        prog="benchmarks", description="Measure what matchlab costs."
    )
    parser.add_argument(
        "--profile",
        default="standard",
        choices=PROFILES,
        help="How much to measure. Default: standard.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Runs per case; the median is reported. Default: 3 for quick, else 1.",
    )
    parser.add_argument(
        "--store",
        default=Store.FILE,
        type=Store,
        choices=list(Store),
        help="Where artifacts go. 'memory' moves the store into peak RSS.",
    )
    parser.add_argument(
        "--only",
        type=Topology,
        choices=list(Topology),
        default=None,
        help="Measure one plan shape and skip the rest.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=measure.DEFAULT_TIMEOUT,
        help="Seconds one case may take before it is treated as wedged.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(RESULTS),
        help=f"Where to write the run's JSON. Default: {RESULTS}.",
    )
    return parser.parse_args(argv)


def environment(arguments: argparse.Namespace, repeats: int) -> Environment:
    """Record the machine and build this run happened on."""
    startup, baseline_rss = measure.baseline()
    return Environment(
        started_at=report.now(),
        profile=arguments.profile,
        repeats=repeats,
        store=arguments.store,
        matchlab_version=_version(),
        git_sha=_git_sha(),
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.machine()}",
        cpus=os.cpu_count() or 0,
        startup_s=startup,
        baseline_rss_bytes=baseline_rss,
    )


def _version() -> str:
    """The installed matchlab version, or a placeholder if it cannot be read."""
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        return version("matchlab")
    except PackageNotFoundError:
        return "unknown"


def _git_sha() -> str:
    """The commit this run measured, short. `unknown` outside a checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return completed.stdout.strip() + ("-dirty" if _dirty() else "")


def _dirty() -> bool:
    """Whether the checkout has uncommitted changes, so a SHA cannot be trusted."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return bool(completed.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    """Run the suite and print what it found.

    Returns:
        `0` if every case was measured, `1` if any failed. A failed case is reported
        and the run carries on.
    """
    arguments = parse(argv)
    console = report.console()

    repeats = arguments.repeats or (3 if arguments.profile == "quick" else 1)
    sweeps = profile(arguments.profile, arguments.store)
    if arguments.only is not None:
        sweeps = _restrict(sweeps, arguments.only)
        if not sweeps:
            console.print(
                f"[red]No case in '{arguments.profile}' uses {arguments.only}."
            )
            return 1

    console.print()
    with console.status("Priming...") as status:
        details = environment(arguments, repeats)

        def generating(warehouse: Warehouse) -> None:
            status.update(
                f"Generating {warehouse.expected_rows:,} rows — "
                f"{warehouse.n_entities:,} entities across "
                f"{warehouse.n_sources} sources"
            )

        built = measure.prepare(sweeps, on_start=generating)

    total = sum(len(sweep.cases) for sweep in sweeps)
    console.print(
        f"[dim]{total} cases, {len(built)} warehouses, {repeats} repeat(s)[/]"
    )

    measured: list[SweepResult] = []
    failures: list[CaseFailed] = []
    done = 0

    for sweep in sweeps:
        results = []
        for case in sweep.cases:
            done += 1
            path, rows = built[case.warehouse]
            console.print(f"[dim]({done}/{total})[/dim] {case.label}", end=" ")
            try:
                result = measure.run(
                    case,
                    path,
                    rows,
                    repeats=repeats,
                    timeout=arguments.timeout,
                )
            except CaseFailed as failure:
                failures.append(failure)
                console.print("[red]failed[/red]")
                continue
            results.append(result)
            console.print(
                f"[green]{result.middle('collect_cold_s'):.2f}s[/green] "
                f"[dim]{format_bytes(int(result.middle('peak_rss_bytes')))} peak[/dim]"
            )

        if results:
            measured.append(
                SweepResult(
                    title=sweep.title,
                    holding=sweep.holding,
                    column=sweep.column,
                    results=tuple(results),
                )
            )

    if not measured:
        _report_failures(failures, console)
        return 1

    run = Run(environment=details, sweeps=tuple(measured))

    report.header(run, console)
    for sweep_result in run.sweeps:
        report.table(sweep_result, details.baseline_rss_bytes, console)
    report.summarise(run, console)

    written = report.save(run, arguments.results)
    console.print(f"[dim]Written to {written}[/dim]\n")

    _report_failures(failures, console)
    return 1 if failures else 0


def _restrict(sweeps: list, topology: Topology) -> list:
    """Keep only the cases using one topology, dropping sweeps left with none."""
    kept = []
    for sweep in sweeps:
        cases = tuple(case for case in sweep.cases if case.topology is topology)
        if cases:
            kept.append(sweep.model_copy(update={"cases": cases}))
    return kept


def _report_failures(failures: list[CaseFailed], console: Console) -> None:
    """Print every case that did not produce a measurement, with its stderr."""
    for failure in failures:
        console.rule(f"[red]{failure.label}")
        console.print(failure.detail.strip() or "no output")
        console.print()


if __name__ == "__main__":
    sys.exit(main())
