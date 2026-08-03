# Benchmarks

What matchlab costs, and under what circumstances.

```shell
just bench                          # the standard profile
just bench --profile quick          # a smoke test, under a minute
just bench --profile full           # every axis pushed
just bench --only mesh --repeats 3  # one shape, three times
```

Each run prints one table per axis and finishes with a summary in words. It also writes the whole run — every repeat, not just the medians shown — to `benchmarks/results/<timestamp>-<profile>.json`, so two runs can be diffed. That directory is gitignored, so committing a run is a deliberate act.

## The four axes

| Axis | Varies | Question |
|---|---|---|
| Plan topology | `dedupe`, `hub`, `chain`, `mesh` | what does the *shape* of a plan cost? |
| Number of entities | 10k → 1M | does cost scale with how much there is to resolve? |
| Records per entity | 1 → 12 | does cost scale with duplication? |
| Number of sources | 2 → 8 | what does a link count that grows quadratically cost? |

Sweeps vary **one axis at a time** from a baseline, rather than crossing every axis with every other. A full cross product is hours of machine time for numbers nobody reads, and a table where two things moved at once cannot answer a question.

## The four topologies

All four match identically — `NaiveDeduper` and `DeterministicLinker` on a cleaned name and postcode — so what varies between them is the plan, not the matcher.

| | Shape | Isolates |
|---|---|---|
| `dedupe` | one source, one model, one resolver | the floor: no link, no layer |
| `hub` | dedupe all, then link the hub to each spoke | a star: `n-1` links, one apex |
| `chain` | add one source per level, on the resolution below | depth |
| `mesh` | dedupe all, then link every pair | breadth: `n(n-1)/2` links |

`hub` is not a cheaper way of writing `mesh`. It cannot merge two spokes that share an entity the hub has never seen, so over the same rows it resolves *more* entities. The tables report entity count per case so that trade is visible rather than assumed.

## Reading a table

| Column | |
|---|---|
| `rows` | rows the plan actually read — `dedupe` reads one source, not all four |
| `steps` | nodes in the plan |
| `cold` | collecting the whole plan into an empty store |
| `µs/row` | the same, per row. **Flat means linear scaling.** |
| `read` | `entities()` — reading the complete resolution back |
| `warm` | collecting an unchanged plan again: every step a cache hit |
| `peak` | high-water resident memory of the measuring process |
| `B/row` | the same, above the idle floor, per row |
| `store` | what the store weighs afterwards |
| `entities` | how many entities came out |

Both floors — interpreter startup and the memory an idle process holds before reading anything — are priced at the start of a run, printed in the header, and kept out of `B/row`.

## How it is measured

**One subprocess per case.** A high-water resident set is monotonic within a process, so a second case measured in the first one's process inherits its peak and reports a figure belonging to neither. Isolation also stops a warmed allocator, an open DuckDB connection or a populated cache from making whichever case ran second look faster than it is.

Peak memory is `getrusage(RUSAGE_SELF).ru_maxrss`, not `tracemalloc`: almost none of the memory here is Python's — it is DuckDB's buffer pool, Polars' frames and Arrow's buffers, and `tracemalloc` is blind to all three.

**Warehouses are generated in the parent and cached on disk**, keyed by their parameters and a generation number. One warehouse serves every topology at a given size, and the generator's own allocations never land in a measured child's peak.

## Why the data looks the way it does

matchlab content-addresses records: two rows with identical field values share a leaf cluster before any model runs. So the generator renders **every row of an entity differently** — case, suffix, spacing — and cleans every rendering back to the same value. Render an entity the same way in two sources and they arrive already merged, the linkers have nothing to do, and every topology returns the same answer at the same price.

That is why this does not use `matchlab.testkit`. The testkit is the right tool for correctness — planted `TrueEntity`s, `diff_resolution`, `PerfectDeduper` — and the wrong one for scale, since it generates entity by entity through Faker. What a benchmark needs from ground truth is much weaker: a `_truth` column, so a case can check it resolved something sane before reporting how fast it did it.

Every case asserts its resolved entity count against what its shape guarantees, and fails loudly rather than reporting a fast wrong answer. `test/test_benchmarks.py` covers the same ground in CI at a few thousand rows.

## Adding to it

- **A topology**: add a member to `Topology` and a builder in `topologies.py`, then say what it must resolve in `Case.expected_entities`.
- **A sweep**: add a function in `cases.py` and reference it from `profile()`.
- **A metric**: add a field to `Measurement`, record it in `run_case.measure`, add a column in `report.table`.
- **Changing the generated values**: bump `data.GENERATION`, or cached warehouses from before the change will be reused silently.
