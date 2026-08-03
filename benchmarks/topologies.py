"""Four plan shapes over the same data, so shape is the only thing that varies.

Every topology here uses `NaiveDeduper` and `DeterministicLinker` on the same two
cleaned fields. That is deliberate: exact matching is close to the cheapest thing
matchlab can be asked to do, so what the timings show is the cost of the *plan* —
hashing, storing, reading identifiers back, materialising resolutions, layering one on
another — rather than the cost of a clever matcher, which would swamp it.

The four shapes and what each isolates:

| | Shape | Isolates |
|---|---|---|
| `DEDUPE` | one source, one model, one resolver | the floor: no link, no layer |
| `HUB` | dedupe all, then link the hub to each spoke | a star: `n-1` links, one apex |
| `CHAIN` | add one source per level, on top of the last resolution | depth |
| `MESH` | dedupe all, then link every pair | breadth: `n(n-1)/2` links, one apex |

`HUB` and `MESH` are not two ways of writing the same plan. A hub cannot merge two
spokes that share an entity the hub has never heard of, so over the same rows it
resolves *more* entities. That is a real property of the shape, which is why resolved
entity count is reported per case and never asserted equal between them.
"""

from collections.abc import Callable, Sequence
from enum import StrEnum
from itertools import combinations
from pathlib import Path
from typing import cast

from sqlalchemy import create_engine

from matchlab import Model, RelationalDBLocation, Resolver, Source, View
from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.linkers import DeterministicLinker

#: Strip case, whitespace, punctuation and company suffixes, so the cosmetic variation
#: the generator plants is absorbed and matching has something to agree on.
NAME = (
    "regexp_replace(upper({0}), '\\s+|\\.|\\bLIMITED\\b|\\bLTD\\b|\\bPLC\\b', '', 'g')"
)

#: Strip case and whitespace only. Postcodes are stable per entity by construction.
POSTCODE = "regexp_replace(upper({0}), '\\s+', '', 'g')"

#: What both matchers agree on. Named once, so no topology can quietly match
#: differently from another and make the comparison between them meaningless.
UNIQUE_FIELDS = ["name", "postcode"]
COMPARISON = "l.name = r.name and l.postcode = r.postcode"


class Topology(StrEnum):
    """The plan shapes the suite knows how to build."""

    DEDUPE = "dedupe"
    HUB = "hub"
    CHAIN = "chain"
    MESH = "mesh"

    @property
    def min_sources(self) -> int:
        """The fewest sources this shape needs to mean anything."""
        return 1 if self is Topology.DEDUPE else 2

    @property
    def summary(self) -> str:
        """One clause naming the shape, for a table header to carry."""
        return {
            Topology.DEDUPE: "one source, deduplicated",
            Topology.HUB: "hub linked to each spoke",
            Topology.CHAIN: "one source added per level",
            Topology.MESH: "every pair linked",
        }[self]


def declare(path: Path, names: Sequence[str]) -> list[Source]:
    """Declare one source per table in a generated warehouse.

    One engine for all of them, because they are one database and a connection per
    source would measure SQLite's connection handling rather than matchlab's.
    """
    location = RelationalDBLocation(
        name="warehouse", client=create_engine(f"sqlite:///{path}")
    )
    return [
        Source(
            location=location,
            name=name,
            extract_transform=f"select key, name, postcode from {name}",
            key_field="key",
        )
        for name in names
    ]


def _cleaning(sources: Sequence[Source]) -> dict[str, str]:
    """Cleaning expressions that work across however many sources a view reads.

    A multi-source view concatenates the extracts diagonally, so each source's columns
    are null on every other source's rows. `coalesce` over the qualified names is what
    folds them back into one column per field. With a single source it degenerates to
    `coalesce(crn_name)`, which costs nothing and keeps one code path.
    """

    def across(field: str) -> str:
        # `Source.f` answers with a list when asked for several fields, so the single
        # -field call has to be narrowed back to the string it actually returns.
        names = [cast("str", source.f(field)) for source in sources]
        return f"coalesce({', '.join(names)})"

    return {
        "name": NAME.format(across("name")),
        "postcode": POSTCODE.format(across("postcode")),
    }


def _view(sources: Sequence[Source], resolver: Resolver | None = None) -> View:
    """A cleaned view of these sources, optionally read through a resolution.

    Passing a resolver is the layering move: `id` becomes the entity that resolver
    assigned rather than the record, so whatever is built on top compares entities.
    """
    return View(*sources, resolver=resolver, cleaning=_cleaning(sources))


def _dedupe(view: View) -> Model:
    """Deduplicate a cleaned view: same cleaned name and postcode is one entity."""
    return view.dedupe(
        model_class=NaiveDeduper, model_settings={"unique_fields": UNIQUE_FIELDS}
    )


def _link(left: View, right: View) -> Model:
    """Link two cleaned views, on the same rule the deduper uses."""
    return left.link(
        right,
        model_class=DeterministicLinker,
        model_settings={"comparisons": COMPARISON},
    )


def build(topology: Topology, sources: Sequence[Source]) -> Resolver:
    """Build a plan of this shape over these sources.

    Args:
        topology: Which shape to build.
        sources: The sources to build it over, hub first. `DEDUPE` uses only the first.

    Returns:
        The apex resolver. Nothing has run: a plan is lazy until it is collected.

    Raises:
        ValueError: If there are too few sources for the shape.
    """
    if len(sources) < topology.min_sources:
        raise ValueError(
            f"{topology} needs at least {topology.min_sources} source(s), "
            f"got {len(sources)}."
        )

    match topology:
        case Topology.DEDUPE:
            return _dedupe(_view(sources[:1])).resolve()
        case Topology.HUB:
            return _star(sources, pairs=_spokes)
        case Topology.MESH:
            return _star(sources, pairs=_every_pair)
        case Topology.CHAIN:
            return _chain(sources)


#: Which links a star-shaped plan builds, given its views.
Pairs = Callable[[Sequence[View]], list[tuple[View, View]]]


def _spokes(views: Sequence[View]) -> list[tuple[View, View]]:
    """Every (hub, spoke) pair. `n-1` links, and the hub in all of them."""
    return [(views[0], spoke) for spoke in views[1:]]


def _every_pair(views: Sequence[View]) -> list[tuple[View, View]]:
    """Every pair of views. `n(n-1)/2` links, and no view privileged."""
    return list(combinations(views, 2))


def _star(sources: Sequence[Source], pairs: Pairs) -> Resolver:
    """Dedupe everything into one resolution, then link the pairs `pairs` chooses.

    `HUB` and `MESH` differ only in which pairs get linked, so they share everything
    else: one dedupe per source collapsed into a single resolution, every source read
    back through it — each view built once and structurally shared by every link that
    uses it — and one apex resolver over all the links.
    """
    dedupes = [_dedupe(_view([source])) for source in sources]
    deduped = dedupes[0].resolve(*dedupes[1:])

    views = [_view([source], resolver=deduped) for source in sources]
    links = [_link(left, right) for left, right in pairs(views)]
    return links[0].resolve(*links[1:])


def _chain(sources: Sequence[Source]) -> Resolver:
    """Add one source per level, each level built on the resolution below it.

    Where `MESH` is one resolution over many links, this is many resolutions stacked:
    `n-1` levels, each reading every source seen so far back through the level below.
    Each level's resolver takes two models — the new source's dedupe, and the link
    joining it to everything already resolved — so every source is still deduplicated
    exactly once and the shapes stay comparable.
    """
    resolved = _dedupe(_view(sources[:1])).resolve()

    for position, source in enumerate(sources[1:], start=1):
        # One view of the new source, shared by both models below: the dedupe that
        # collapses its own duplicates, and the link that attaches it to the rest.
        arriving = _view([source])
        accumulated = _view(sources[:position], resolver=resolved)
        resolved = _link(accumulated, arriving).resolve(_dedupe(arriving))

    return resolved
