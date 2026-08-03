"""Say what the run found."""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from benchmarks.cases import Result, Run, SweepResult
from benchmarks.topologies import Topology
from matchlab.adapters.base import format_bytes

#: The narrowest a run may be rendered. The tables carry eleven columns because
#: eleven things are worth comparing side by side, and wrapping them into a
#: two-line-per-row stack makes the comparison unreadable
MIN_WIDTH = 118


def console() -> Console:
    """A console wide enough for the tables, whatever the terminal thinks."""
    return Console(width=max(shutil.get_terminal_size().columns, MIN_WIDTH))


def header(run: Run, console: Console) -> None:
    """Print what this run was, and on what."""
    environment = run.environment
    console.print()
    console.rule(f"[bold]matchlab benchmarks — {environment.profile}")

    facts = [
        ("matchlab", f"{environment.matchlab_version} ({environment.git_sha})"),
        ("python", f"{environment.python} on {environment.platform}"),
        ("cpus", str(environment.cpus)),
        ("store", str(environment.store)),
        ("repeats", f"{environment.repeats} (median reported)"),
        (
            "per-case floor",
            f"{environment.startup_s:.2f}s startup, "
            f"{format_bytes(environment.baseline_rss_bytes)} resident — "
            "excluded from B/row",
        ),
    ]
    width = max(len(name) for name, _ in facts)
    for name, value in facts:
        console.print(f"  [dim]{name:>{width}}[/dim]  {value}")
    console.print()


def table(sweep: SweepResult, baseline_rss: int, console: Console) -> None:
    """Print one sweep as one table."""
    rendered = Table(
        title=f"[bold]{sweep.title}[/bold]  [dim]— holding {sweep.holding}[/dim]",
        title_justify="left",
        header_style="bold",
        expand=False,
    )
    for name, justify in (
        (sweep.column, "left"),
        ("rows", "right"),
        ("steps", "right"),
        ("cold", "right"),
        ("µs/row", "right"),
        ("read", "right"),
        ("warm", "right"),
        ("peak", "right"),
        ("B/row", "right"),
        ("store", "right"),
        ("entities", "right"),
    ):
        rendered.add_column(name, justify=justify, no_wrap=True)

    for result in sweep.results:
        rendered.add_row(
            result.axis(sweep.column),
            f"{result.rows:,}",
            str(result.steps),
            f"{result.middle('collect_cold_s'):.2f}s",
            f"{result.us_per_row:.1f}",
            f"{result.middle('entities_s'):.2f}s",
            f"{result.middle('collect_warm_s'):.2f}s",
            format_bytes(int(result.middle("peak_rss_bytes"))),
            f"{result.bytes_per_row(baseline_rss):,.0f}",
            format_bytes(int(result.middle("store_bytes"))),
            f"{result.runs[0].n_entities_resolved:,}",
        )

    console.print(rendered)
    console.print()


def summarise(run: Run, console: Console) -> None:
    """State what the tables jointly show, in sentences.

    Where a profile did not measure something, the claim is omitted.
    """
    console.rule("[bold]What this run found")
    console.print()

    lines = [
        *_scaling(run, "Number of entities", "entities"),
        *_scaling(run, "Records per entity", "duplicate records"),
        *_topologies(run),
        *_caching(run),
    ]

    if not lines:
        console.print("  [dim]Nothing to summarise: no sweep had enough cases.[/dim]\n")
        return

    for line in lines:
        # Padded rather than prefixed, so a claim that wraps stays inside its bullet
        # instead of falling back to the left margin and reading as a new one.
        console.print(Padding(Text("• ", style="dim").append_text(line), (0, 2, 1, 2)))


def _scaling(run: Run, title: str, axis: str) -> list[Text]:
    """How time and memory move as one size axis grows.

    Both claims compare the largest case with the smallest — end to end and in that
    direction. A range taken as min-to-max would read the same whether the cost per
    row rose or fell, which is the only thing either sentence is trying to say.
    """
    sweep = run.find(title)
    if sweep is None or len(sweep.results) < 2:
        return []

    baseline_rss = run.environment.baseline_rss_bytes
    first, last = sweep.results[0], sweep.results[-1]
    rows = last.rows / first.rows

    seconds = _growth(last.middle("collect_cold_s"), first.middle("collect_cold_s"))
    resident = _growth(
        last.middle("peak_rss_bytes") - baseline_rss,
        first.middle("peak_rss_bytes") - baseline_rss,
    )

    return [
        Text.from_markup(
            f"[bold]{rows:.0f}x the rows[/bold] by varying {axis} took "
            f"[bold]{seconds:.1f}x the time[/bold] — {_shape(seconds, rows)}. "
            f"Per row that is {first.us_per_row:.1f} µs at {first.rows:,} rows "
            f"against {last.us_per_row:.1f} µs at {last.rows:,}."
        ),
        Text.from_markup(
            f"The same growth took [bold]{resident:.1f}x the memory[/bold] above the "
            f"{format_bytes(baseline_rss)} an idle process already holds — "
            f"{_shape(resident, rows)}. Per row that is "
            f"{first.bytes_per_row(baseline_rss):,.0f} B against "
            f"{last.bytes_per_row(baseline_rss):,.0f} B."
        ),
    ]


def _growth(after: float, before: float) -> float:
    """How many times over something grew. Guards the zero a fast case can produce."""
    return after / before if before > 0 else float("inf")


def _shape(observed: float, driver: float) -> str:
    """Name a growth ratio against the growth in data that drove it.

    Sublinear is the good answer and the common one at these sizes: fixed overheads —
    opening a store, importing, priming a connection — are a smaller share of a larger
    run, so the per-row cost falls as the data grows.
    """
    ratio = observed / driver if driver else observed
    if ratio < 0.9:
        return "sublinear, so the per-row cost falls as the data grows"
    if ratio <= 1.25:
        return "linear"
    if ratio <= 2.5:
        return "superlinear, so the per-row cost rises with the data"
    return "steeply superlinear, which is a scaling problem"


def _topology_line(topology: Topology, result: Result, floor: Result) -> Text:
    """State one topology's cost against `dedupe`, in rows grown vs. time grown."""
    rows = _growth(result.rows, floor.rows)
    seconds = _growth(result.middle("collect_cold_s"), floor.middle("collect_cold_s"))
    return Text.from_markup(
        f"[bold]{topology}[/bold] read [bold]{rows:.1f}x the rows[/bold] dedupe did "
        "(every source, against dedupe's one) — "
        f"[bold]{seconds:.1f}x the time[/bold] for that extra volume, "
        f"{_shape(seconds, rows)}. {result.steps} steps, "
        f"{result.middle('collect_cold_s'):.2f}s for {result.rows:,} rows."
    )


def _topologies(run: Run) -> list[Text]:
    """What each plan shape costs, against the cheapest one that exists.

    `DEDUPE` reads one source; the other three read every source, so they always cover
    more rows in the same sweep. Comparing raw µs/row against that floor is not a fair
    reading — a shape landing on *more* data amortises the fixed cost of opening a
    store and connecting to the warehouse over more rows, and looks artificially
    cheaper for it, regardless of how much the shape itself adds. Comparing time
    growth against row growth — the same move `_scaling` makes across a size sweep —
    is what actually isolates the shape's cost from the volume difference.
    """
    sweep = run.find("Plan topology")
    if sweep is None:
        return []

    by_topology = {result.case.topology: result for result in sweep.results}
    floor = by_topology.get(Topology.DEDUPE)
    if floor is None:
        return []

    lines = [
        _topology_line(topology, result, floor)
        for topology, result in by_topology.items()
        if topology is not Topology.DEDUPE
    ]

    hub, mesh = by_topology.get(Topology.HUB), by_topology.get(Topology.MESH)
    if hub is not None and mesh is not None:
        extra = hub.runs[0].n_entities_resolved - mesh.runs[0].n_entities_resolved
        lines.append(
            Text.from_markup(
                f"hub left [bold]{extra:,} entities unmerged[/bold] that mesh joined "
                f"({hub.runs[0].n_entities_resolved:,} against "
                f"{mesh.runs[0].n_entities_resolved:,}). A star cannot reach two "
                "spokes that share something the hub never saw, so whatever it saves "
                "it pays for in the answer."
            )
            if extra > 0
            else Text.from_markup(
                "hub and mesh agreed on "
                f"[bold]{mesh.runs[0].n_entities_resolved:,} entities[/bold]. At this "
                "overlap the hub can reach everything, so its cheaper shape costs "
                "nothing — which is not true at lower coverage."
            )
        )
    return lines


def _caching(run: Run) -> list[Text]:
    """What a second collect over an unchanged plan costs."""
    results = [result for sweep in run.sweeps for result in sweep.results]
    ratios = [
        result.middle("collect_warm_s") / result.middle("collect_cold_s")
        for result in results
        if result.middle("collect_cold_s") > 0
    ]
    if not ratios:
        return []

    return [
        Text.from_markup(
            f"Re-collecting an unchanged plan took "
            f"[bold]{min(ratios):.1%}–{max(ratios):.1%} of the cold run[/bold]. "
            "That is the price of proving nothing changed, and it is what you pay on "
            "every iteration after the first."
        )
    ]


def save(run: Run, directory: Path) -> Path:
    """Write the whole run to a timestamped JSON file, and return where it went.

    Every repeat is kept, not just the medians the tables showed: the file exists to be
    diffed against a later run, and a spread that has been averaged away cannot say
    whether a difference is a regression or a noisy laptop.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp = run.environment.started_at.replace(":", "-")
    path = directory / f"{stamp}-{run.environment.profile}.json"
    path.write_text(json.dumps(json.loads(run.model_dump_json()), indent=2) + "\n")
    return path


def now() -> str:
    """A UTC timestamp, to the second, safe in a filename."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
