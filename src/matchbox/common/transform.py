import logging
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Generic, Hashable, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rustworkx as rx
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from matchbox.client.results import ClusterResults, ProbabilityResults
from matchbox.common.hash import (
    IntMap,
    list_to_value_ordered_hash,
)

T = TypeVar("T", bound=Hashable)

logic_logger = logging.getLogger("mb_logic")

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


def to_clusters(results: ProbabilityResults) -> ClusterResults:
    """
    Converts probabilities into a list of connected components formed at each threshold.

    Returns:
        ClusterResults sorted by threshold descending.
    """
    G = rx.PyGraph()
    added: dict[bytes, int] = {}
    components: dict[str, list] = {"parent": [], "child": [], "threshold": []}

    # Sort probabilities descending and group by probability
    edges_df = (
        results.dataframe.sort_values("probability", ascending=False)
        .filter(["left_id", "right_id", "probability"])
        .astype(
            {"left_id": "large_binary[pyarrow]", "right_id": "large_binary[pyarrow]"}
        )
    )

    # Get unique probability thresholds, sorted
    thresholds = sorted(edges_df["probability"].unique())

    # Process edges grouped by probability threshold
    for prob in thresholds:
        threshold_edges = edges_df[edges_df["probability"] == prob]
        # Get state before adding this batch of edges
        old_components = {frozenset(comp) for comp in rx.connected_components(G)}

        # Add all nodes and edges at this probability threshold
        edge_values = threshold_edges[["left_id", "right_id"]].values
        for left, right in edge_values:
            for hash_val in (left, right):
                if hash_val not in added:
                    idx = G.add_node(hash_val)
                    added[hash_val] = idx

            G.add_edge(added[left], added[right], None)

        new_components = {frozenset(comp) for comp in rx.connected_components(G)}
        changed_components = new_components - old_components

        # For each changed component, add ALL members at current threshold
        for comp in changed_components:
            children = sorted([G.get_node_data(n) for n in comp])
            parent = list_to_value_ordered_hash(children)

            components["parent"].extend([parent] * len(children))
            components["child"].extend(children)
            components["threshold"].extend([prob] * len(children))

    return ClusterResults(
        dataframe=DataFrame(components).convert_dtypes(dtype_backend="pyarrow"),
        model=results.model,
        metadata=results.metadata,
    )


def attach_components_to_probabilities(probabilities: pa.Table) -> pa.Table:
    """
    Takes an Arrow table of probabilities and adds a component column.

    Expects an Arrow table of column, left, right, probability.

    Returns a table with an additional column, component.
    """
    # Create index to use in graph
    unique = pc.unique(
        pa.concat_arrays(
            [
                probabilities["left"].combine_chunks(),
                probabilities["right"].combine_chunks(),
            ]
        )
    )
    left_indices = pc.index_in(probabilities["left"], unique)
    right_indices = pc.index_in(probabilities["right"], unique)

    # Create and process graph
    n_nodes = len(unique)
    n_edges = len(probabilities)

    graph = rx.PyGraph(node_count_hint=n_nodes, edge_count_hint=n_edges)
    graph.add_nodes_from(range(n_nodes))

    edges = tuple(zip(left_indices.to_numpy(), right_indices.to_numpy(), strict=True))
    graph.add_edges_from_no_data(edges)

    components = rx.connected_components(graph)

    # Convert components to arrays, map back to input to join, and reattach
    component_indices = np.concatenate([np.array(list(c)) for c in components])
    component_labels = np.repeat(
        np.arange(len(components)), [len(c) for c in components]
    )

    node_to_component = np.zeros(len(unique), dtype=np.int64)
    node_to_component[component_indices] = component_labels

    edge_components = pa.array(node_to_component[left_indices.to_numpy()])

    return probabilities.append_column("component", edge_components).sort_by(
        [("component", "ascending"), ("probability", "descending")]
    )


class DisjointSet(Generic[T]):
    """
    Disjoint set forest with "path compression" and "union by rank" heuristics.

    This follows implementation from Cormen, Thomas H., et al. Introduction to
    algorithms. MIT press, 2022
    """

    def __init__(self):
        self.parent: dict[T, T] = {}
        self.rank: dict[T, int] = {}

    def _make_set(self, x: T) -> None:
        self.parent[x] = x
        self.rank[x] = 0

    def union(self, x: T, y: T) -> None:
        self._link(self._find(x), self._find(y))

    def _link(self, x: T, y: T) -> None:
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
        else:
            self.parent[x] = y
            if self.rank[x] == self.rank[y]:
                self.rank[y] += 1

    def _find(self, x: T) -> T:
        if x not in self.parent:
            self._make_set(x)
            return x

        if x != self.parent[x]:
            self.parent[x] = self._find(self.parent[x])

        return self.parent[x]

    def get_components(self) -> list[set[T]]:
        components = defaultdict(set)
        for x in self.parent:
            root = self._find(x)
            components[root].add(x)
        return list(components.values())


def component_to_hierarchy(
    table: pa.Table, salt: int, dtype: pa.DataType = pa.uint64
) -> pa.Table:
    """
    Convert pairwise probabilities into a hierarchical representation.

    Assumes data is pre-sorted by probability descending.

    Args:
        table: Arrow Table with columns ['left', 'right', 'probability']

    Returns:
        Arrow Table with columns ['parent', 'child', 'probability']
    """
    ascending_probs = np.sort(
        pc.unique(table["probability"]).to_numpy(zero_copy_only=False)
    )
    probs = ascending_probs[::-1]

    djs = DisjointSet[int]()  # implements connected components
    im = IntMap(salt=salt)  # generates IDs for new clusters
    current_roots: dict[int, set[int]] = defaultdict(set)  # tracks ultimate parents
    hierarchy: list[tuple[int, int, float]] = []  # the output of this function

    for threshold in probs:
        # Get current probability rows
        mask = pc.equal(table["probability"], threshold)
        current_probs = table.filter(mask)

        # Add new pairwise relationships at this threshold
        for left, right in zip(
            current_probs["left"].to_numpy(),
            current_probs["right"].to_numpy(),
            strict=True,
        ):
            djs.union(left, right)
            parent = im.index(left, right)
            hierarchy.extend([(parent, left, threshold), (parent, right, threshold)])
            current_roots[left].add(parent)
            current_roots[right].add(parent)

        for children in djs.get_components():
            if len(children) <= 2:
                continue  # Skip pairs already handled by pairwise probabilities

            if im.has_mapping(*children):
                continue  # Skip unchanged components from previous thresholds

            parent = im.index(*children)
            prev_roots: set[int] = set()
            for child in children:
                prev_roots.update(current_roots[child])
                current_roots[child] = {parent}

            for r in prev_roots:
                hierarchy.append((parent, r, threshold))

    parents, children, probs = zip(*hierarchy, strict=True)
    return pa.table(
        {
            "parent": pa.array(parents, type=dtype()),
            "child": pa.array(children, type=dtype()),
            "probability": pa.array(probs, type=pa.uint8()),
        }
    )


def to_hierarchical_clusters(
    probabilities: pa.Table,
    proc_func: Callable[[pa.Table, pa.DataType], pa.Table] = component_to_hierarchy,
    dtype: pa.DataType = pa.int64,
    timeout: int = 300,
) -> pa.Table:
    """
    Converts a table of pairwise probabilities into a table of hierarchical clusters.

    Args:
        probabilities: Arrow table with columns ['component', 'left', 'right',
            'probability']
        proc_func: Function to process each component
        timeout: Maximum seconds to wait for each component to process

    Returns:
        Arrow table with columns ['parent', 'child', 'probability']
    """
    console = Console()
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    probabilities = probabilities.sort_by([("component", "ascending")])
    components = pc.unique(probabilities["component"])
    n_cores = multiprocessing.cpu_count()
    n_components = len(components)

    logic_logger.info(f"Processing {n_components:,} components using {n_cores} workers")

    # Split table into separate component tables
    component_col = probabilities["component"]
    indices = []
    start_idx = 0

    with Progress(*progress_columns, console=console) as progress:
        split_task = progress.add_task(
            "[cyan]Splitting tables...", total=len(component_col)
        )

        for i in range(1, len(component_col)):
            if component_col[i] != component_col[i - 1]:
                indices.append((start_idx, i))
                start_idx = i
            progress.update(split_task, advance=1)

        indices.append((start_idx, len(component_col)))
        progress.update(split_task, completed=len(component_col))

    component_tables = []
    for start, end in indices:
        idx_array = pa.array(range(start, end))
        component_tables.append(probabilities.take(idx_array))

    # Process components in parallel
    results = []
    with Progress(*progress_columns, console=console) as progress:
        process_task = progress.add_task(
            "[green]Processing components...", total=len(component_tables)
        )

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = [
                executor.submit(proc_func, component_table, salt=salt, dtype=dtype)
                for salt, component_table in enumerate(component_tables)
            ]

            for future in futures:
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                    progress.update(process_task, advance=1)
                except TimeoutError:
                    logic_logger.error(
                        f"Component processing timed out after {timeout} seconds"
                    )
                    raise
                except Exception as e:
                    logic_logger.error(f"Error processing component: {str(e)}")
                    raise

    logic_logger.info(f"Completed processing {len(results):,} components successfully")

    # Create empty table if no results
    if not results:
        logic_logger.warning("No results to concatenate")
        return pa.table(
            {
                "parent": pa.array([], type=dtype()),
                "child": pa.array([], type=dtype()),
                "probability": pa.array([], type=pa.uint8()),
            }
        )

    return pa.concat_tables(results)
