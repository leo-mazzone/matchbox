"""Build a plan one step at a time.

The same warehouse as `with_matchlab.py`, which builds its plan in one function and
collects it once. This one adds a single step per beat and stops to look at what that
step produced, so you watch a plan grow rather than read one finished.

Written to introduce matchlab to someone who has never seen it, and to reintroduce it
to someone who knew Matchbox. Notes marked `Matchbox →` say what the old API called
the same thing; skip them if you never used it.

    python warehouse.py        # once, to build the SQLite warehouse
    python step_by_step.py

It pauses between beats at a terminal. `--no-pause` runs it straight through, and
`--store run.duckdb` keeps the store on disk so a second run is entirely cache.

Lines marked `?` are real questions, not rhetoric. The design choices around them are
the ones we are least sure about, and what you think on first meeting them is the
useful thing — that reaction is not recoverable once you're used to the library. They
are collected again at the end so you can go back through them.
"""

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import polars as pl
from sqlalchemy import create_engine

from matchlab import RelationalDBLocation, Source, set_default_adapter
from matchlab.adapters import DuckDBAdapter
from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.linkers import DeterministicLinker

DB = Path(__file__).parent / "warehouse.sqlite"

# One cleaning rule each, applied to whichever column a source happens to use. `{0}`
# is filled with the source-qualified column name — see beat 2.
NAME = (
    "regexp_replace(upper({0}), '\\s+|\\.|\\bLIMITED\\b|\\bLTD\\b|\\bPLC\\b', '', 'g')"
)
POSTCODE = "regexp_replace(upper({0}), '\\s+', '', 'g')"

QUESTIONS: list[str] = []
PAUSE = False


# -- narration ------------------------------------------------------------------------


def wrap(text: str, first: str = "", rest: str = "") -> str:
    """Reflow a triple-quoted paragraph to one width, however it was indented."""
    return textwrap.fill(
        " ".join(text.split()), width=79, initial_indent=first, subsequent_indent=rest
    )


def beat(number: int, title: str) -> None:
    """Start a beat, pausing first if someone is watching."""
    if PAUSE and number > 1:
        input("\n            ⏎ ")
    print(f"\n\n{'━' * 79}\n {number}. {title}\n{'━' * 79}\n")


def note(text: str) -> None:
    print(wrap(text), "\n")


def matchbox(text: str) -> None:
    """A note only a former Matchbox user needs."""
    print(wrap(text, "  │ ", "  │ "), "\n")


def question(text: str) -> None:
    """Ask for an opinion, and keep it for the summary at the end."""
    QUESTIONS.append(" ".join(text.split()))
    print(wrap(text, "  ? ", "    "), "\n")


def show(label: str, value: object) -> None:
    print(f"{label}\n")
    print(textwrap.indent(str(value), "    "), "\n")


# -- the plan -------------------------------------------------------------------------


def main() -> object:
    """Build the plan a step at a time. Returns the resolver it ends with."""
    from warehouse import truth  # noqa: PLC0415 - lives beside this script

    # ---------------------------------------------------------------------------------
    beat(1, "A source is a query plus a key")

    note("""
        Everything starts from data you already have. A location holds the client; a
        source is one SQL query against it, plus the column that keys the rows. That
        is the whole declaration — no field list, no types, no registration.
    """)

    warehouse = RelationalDBLocation(
        name="warehouse", client=create_engine(f"sqlite:///{DB}")
    )

    crn = Source(
        location=warehouse,
        name="crn",
        extract_transform=(
            "select company_number, company_name, registered_postcode from crn"
        ),
        key_field="company_number",
    )

    show(
        "crn.sample(3)  — peek at the warehouse, without collecting anything:",
        crn.sample(3),
    )

    note("""
        `name` is not a way of finding this source later. It prefixes every column the
        source contributes and tags its rows in a result, so it is part of the output.
        `crn.f("company_name")` is how you spell that prefix without hardcoding it:
    """)

    show("crn.f('company_name')", crn.f("company_name"))

    note("""
        Nothing has run. `crn` is a lazy step — the first node of a plan — and you can
        already draw it. `○` is a step that hasn't run; the number in brackets is its
        position, which is the same number the logs and `draw()` use all the way
        through.
    """)

    show("crn.draw()", crn.draw())

    note("""
        `collect()` is the only thing in this file that does work. Call it on any node
        and it walks that node's inputs first, running what isn't already stored. For
        a source, that means reading the warehouse and content-addressing every row.
    """)

    crn.collect(interactive=False)
    show("crn.draw()  — same node, now `●`, meaning it ran:", crn.draw())

    matchbox("""
        Matchbox → There is no `DAG` to create and no `dag.source(...)` to register
        with. The step you are holding is the pipeline: each one keeps a reference to
        its inputs, so there is no container object and nothing to keep in sync.
    """)

    matchbox("""
        Matchbox → `index_fields` is gone. Identity is now every column the extract
        returns except the key, so two rows are the same record exactly when the query
        returns the same values for both. Pull a `last_updated` column through and
        every row becomes distinct.
    """)

    question("""
        Does identity-from-the-SELECT read as one less thing to get wrong, or as
        losing a useful knob? Have you got a query where you'd want a column returned
        but excluded from identity?
    """)

    # ---------------------------------------------------------------------------------
    beat(2, "A view is what the model will see")

    note("""
        A model never reads a source directly. It reads a view: which records are in
        scope, and what shape they're in. Cleaning is a clause of that — DuckDB SQL
        per output column, over the source-qualified names.
    """)

    crn_cleaning = {
        "name": NAME.format(crn.f("company_name")),
        "postcode": POSTCODE.format(crn.f("registered_postcode")),
    }
    crn_view = crn.view(cleaning=crn_cleaning)
    crn_view.collect(interactive=False)

    show("crn_view.data()", crn_view.data())

    note("""
        Only the columns you named survive, plus `id` — the thing the model matches
        on. Read a source directly and `id` is one record; read one through a resolver
        and `id` is that resolver's entity, so several records share it. That single
        difference is what layering is, and it turns up in beat 6.
    """)

    note("""
        You can skip `.view()` entirely: `crn.dedupe(...)` inserts one for you, and
        the right-hand side of a link is viewed for you too. Reach for it explicitly
        when you want one of the three things only a view does — clean columns,
        `group=True`, or read through a resolver.
    """)

    matchbox("""
        Matchbox → `source.query(...)` became `source.view(...)`, and `cleaning` is
        keyword-only so that `source.view(cleaning=...)` and
        `resolver.view(source, cleaning=...)` read the same way. It's `view` rather
        than `clean` because saying which records are in scope is the node's job;
        cleaning is optional.
    """)

    question("""
        `view()` passes every column through. `view(cleaning={})` is a real projection
        that selects nothing, leaving only `id`. Deliberately different — "I didn't
        ask for a projection" versus "I asked for an empty one". Too subtle a
        distinction to hang on a pair of braces?
    """)

    # ---------------------------------------------------------------------------------
    beat(3, "A model proposes edges, and stops there")

    note("""
        `NaiveDeduper` says two records match when the fields you name are equal. It
        does not produce groups — it produces scored pairs. Turning pairs into
        entities is somebody else's job.
    """)

    crn_dedupe = crn_view.dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": ["name", "postcode"]},
    )
    crn_dedupe.collect(interactive=False)

    show("crn_dedupe.edges()", crn_dedupe.edges())

    note("""
        Two pairs, because the extract has two near-duplicates that the cleaning made
        identical: a punctuation-and-spacing variant of Acme, and Gamma registered
        once as 'Limited' and once as 'Ltd'. Swapping this for Splink is
        `model_class` and `model_settings` and nothing else — the plan around it does
        not change shape.
    """)

    matchbox("""
        Matchbox → `query.deduper(name=..., ...)` became `.dedupe(...)`, and the name
        is gone. Steps have no names at all now; see beat 4.
    """)

    question("""
        Models emit edges, resolvers turn edges into clusters. It buys you resolving
        several methodologies together with per-model thresholds — but it is two
        concepts where a lot of tools have one. Worth the extra noun on first
        encounter?
    """)

    # ---------------------------------------------------------------------------------
    beat(4, "A resolver turns edges into entities")

    crn_entities = crn_dedupe.resolve()
    crn_entities.collect(interactive=False)

    resolved = crn_entities.entities()
    show("crn_entities.entities()", resolved)

    note("""
        `root` is the entity, `leaf` is the content-addressed record identity, `key`
        is what the source called it. One row per record, materialised when you
        collect — so every query from here on is a read, not a re-derivation.
    """)

    print(
        f"    {resolved['root'].n_unique()} entities from {resolved.height} records\n"
    )

    show("crn_entities.draw()  — the plan so far, four steps:", crn_entities.draw())

    note("""
        `resolve()` defaults to connected components. It takes several models, which
        is how you resolve two methodologies at once and trust a strict one further
        than a loose one, via `resolver_settings={"thresholds": {model: 0.9}}` — keyed
        on the model object, because you are already holding it.
    """)

    matchbox("""
        Matchbox → `model.resolver(name=..., resolver_class=Components)` became
        `.resolve()`. Thresholds take the model itself rather than its name, so a typo
        is a NameError at build time instead of a failure at run time.
    """)

    question("""
        Steps have no names — everything is addressed by position (`[step 2]`), and
        only a finished resolution gets a label, via `publish()` in beat 7. Naming
        every step was busywork, and the names were rarely reused. But positions shift
        when you add a step upstream. Does `[step 2]` still work for you on a plan
        with forty nodes?
    """)

    # ---------------------------------------------------------------------------------
    beat(5, "A second source, and the part that doesn't run twice")

    note("""
        Now the same three steps for a second source, and one resolver over both
        dedupes. Watch the log: seven steps in the plan, and only the four new ones
        run — the `crn` branch is already stored. Steps are addressed by their
        configuration and their inputs' fingerprints, so re-collecting an unchanged
        plan does no work, and adding a step runs only that step.
    """)

    dh = Source(
        location=warehouse,
        name="dh",
        extract_transform="select dh_id, name, postcode from dh",
        key_field="dh_id",
    )
    dh_cleaning = {
        "name": NAME.format(dh.f("name")),
        "postcode": POSTCODE.format(dh.f("postcode")),
    }
    dh_dedupe = dh.view(cleaning=dh_cleaning).dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": ["name", "postcode"]},
    )

    deduped = crn_dedupe.resolve(dh_dedupe)
    deduped.collect(interactive=False)

    note("""
        Now scroll back and read the closing line of all five collects in order. On a
        cold store they read (1 ran, 0 cached), (1 ran, 1 cached), (1 ran, 2 cached),
        (1 ran, 3 cached), (4 ran, 3 cached) — every beat of this file has paid only
        for the step it added. Run it again with `--store run.duckdb` and each of them
        reads `0 ran` instead, in a fresh process. That is the thing to take away:
        building a plan a step at a time is not a teaching device, it is how you are
        meant to work, because the cost of adding a step is the step.
    """)

    show("deduped.draw()", deduped.draw())

    question("""
        A step's cache key comes from its configuration, not its output — so
        reformatting a cleaning expression in a way that changes no data still
        invalidates everything below it. Conservative and predictable, or annoying
        enough that you'd want output-hashing?
    """)

    # ---------------------------------------------------------------------------------
    beat(6, "Linking on top of a resolution")

    note("""
        Two sources are deduplicated but unconnected. To link them we could match raw
        rows against raw rows — but the dedupe already told us Acme is one company in
        `crn`, and it would be strange to make the linker rediscover that. So read
        each source *through* the resolver: same columns, but `id` is now the
        deduplicated entity, and the linker matches entities rather than records.
    """)

    crn_entity_view = deduped.view(crn, cleaning=crn_cleaning)
    dh_entity_view = deduped.view(dh, cleaning=dh_cleaning)

    linked = crn_entity_view.link(
        dh_entity_view,
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.name = r.name and l.postcode = r.postcode"},
    )
    companies = linked.resolve()
    companies.collect(interactive=False)

    show("linked.edges()", linked.edges())

    note("""
        Read the linker's log line above and you'll see it compared 8 rows against 6 —
        still one row per record, not one per entity. Reading through a resolver
        changes what `id` *means*, not how many rows there are, and that is usually
        what you want: two spellings of Acme are two pieces of evidence for the same
        entity. When you'd rather have one row, `group=True` collapses each `id` and
        every cleaning expression becomes an aggregate — `any_value(...)`,
        `list(distinct ...)` — so you say how each column combines.
    """)

    note("""
        Note what did *not* need saying: the dedupe's groupings are still there.
        A resolver always carries its inputs' resolutions forward, so records the link
        never touches keep their upstream grouping instead of quietly reverting to
        singletons. Three link edges, but the answer below has more than three merges
        in it.
    """)

    resolution = companies.entities()
    groups: dict[int, set[str]] = {}
    for row in resolution.iter_rows(named=True):
        groups.setdefault(row["root"], set()).add(row["key"])

    print(f"    {len(groups)} entities from {resolution.height} records\n")

    expected = truth()
    for keys in sorted(groups.values(), key=lambda k: sorted(k)[0]):
        mark = "✓" if len({expected[key] for key in keys}) == 1 else "✗"
        print(f"      {mark} {sorted(keys)}")
    print()

    note("""
        Every group is one real company. Being able to check that at all is why
        `warehouse.py` carries a truth column it never shows the pipeline — and on
        data where you don't have one, that check is what the evaluation tools do
        instead: sample clusters, record judgements, score precision and recall.
    """)

    question("""
        Building the layered views meant passing `crn_cleaning` in a second time,
        because a view read through a resolver is a new node and doesn't inherit the
        earlier one's cleaning. Correct, and a papercut. Would you want the cleaning
        carried forward by default, or is restating it the honest thing?
    """)

    # ---------------------------------------------------------------------------------
    beat(7, "Publishing, and getting your identifiers back")

    note("""
        A plan you never publish still runs — it is just unlabelled, addressed by
        fingerprint. Publishing puts a label on a *collected resolution*, which is
        what makes it findable later, from another process, without this script.
    """)

    companies = companies.publish("companies")

    show(
        "companies.get_lookup()  — your identifiers side by side, a column per source:",
        companies.get_lookup(),
    )

    note("""
        `root` repeats where an entity holds more than one key in a source: the join
        is across sources, so two `crn` keys against one `dh` key is two rows. If you
        want one row per record instead, that is `entities()` from beat 4.
    """)

    show(
        "companies.lookup_key(from_source='crn', to_sources=['dh'], key='crn-01')",
        companies.lookup_key(from_source="crn", to_sources=["dh"], key="crn-01"),
    )

    # The entity that pulled the most records together — the interesting one to look at.
    biggest = max(groups, key=lambda root: len(groups[root]))
    show(
        f"companies.view_entity({biggest})  — the rows behind one entity, as matched:",
        companies.view_entity(biggest, merge_fields=True),
    )

    note("""
        Those reads hit the materialised resolution, not the warehouse — no connection
        needed, and `view_entity` shows the values the matching actually saw rather
        than whatever the source says today.
    """)

    matchbox("""
        Matchbox → publishing replaced naming, and reads moved onto the resolver
        itself: `as_dump()` is `entities()`, `as_lookup()` is `get_lookup()`,
        `view_cluster()` is `view_entity()`. `ResolverMatches` is gone — every one of
        its methods was a projection of a table the resolver already materialises.
    """)

    question("""
        Publishing is an operation on a result rather than a property of a plan, so
        the same plan can be collected without ever being labelled. Does the
        name/label split land, or would you expect to name the pipeline up front?
    """)

    # ---------------------------------------------------------------------------------
    beat(8, "Where to go next")

    note("""
        `with_matchlab.py` is this same pipeline over four sources, written the way
        you'd actually write it: one `build()` function, collected once. Read it next
        — it should now be a shorter read than this file. `by_hand.py` is the same
        answer in Polars and a union-find, and the README compares what each costs.
    """)

    print("  The questions again, if you'd rather answer them in one go:\n")
    for index, text in enumerate(QUESTIONS, start=1):
        print(wrap(text, f"  {index}. ", "     "), "\n")

    return companies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-pause", action="store_true", help="don't wait between beats"
    )
    parser.add_argument(
        "--store",
        default=":memory:",
        help="where to keep artifacts; a path makes a second run all cache",
    )
    args = parser.parse_args()

    if not DB.exists():
        sys.exit(f"No warehouse at {DB}. Run `python warehouse.py` first.")

    PAUSE = sys.stdin.isatty() and not args.no_pause

    # The log is where cached-versus-ran shows up, and it's the point of beat 5. It
    # goes to stdout rather than the default stderr so that the narration and the log
    # stay in order when you pipe this into a pager or a file.
    logging.basicConfig(level=logging.INFO, format="  %(message)s", stream=sys.stdout)

    # One store for the whole session, so every `collect()` shares a cache. At a
    # terminal `collect()` would draw a live tree; these collects pass
    # `interactive=False` so the run leaves a readable record instead.
    set_default_adapter(DuckDBAdapter(args.store))

    pl.Config(
        tbl_rows=20,
        tbl_width_chars=90,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    )

    main()
