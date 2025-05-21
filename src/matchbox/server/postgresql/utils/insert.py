"""Utilities for inserting data into the PostgreSQL backend."""

from collections import defaultdict
from typing import Iterator

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from sqlalchemy import alias, delete, or_, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import ModelResolutionName
from matchbox.common.exceptions import MatchboxResolutionAlreadyExists
from matchbox.common.graph import ResolutionNodeType
from matchbox.common.hash import Cluster, hash_arrow_table
from matchbox.common.logging import logger
from matchbox.common.sources import SourceConfig
from matchbox.common.transform import DisjointSet
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    PKSpace,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    SourceConfigs,
)
from matchbox.server.postgresql.utils.db import (
    compile_sql,
    ingest_to_temporary_table,
    large_ingest,
)


class HashIDMap:
    """An object to help map between IDs and hashes.

    When given a set of IDs, returns their hashes. If any ID doesn't have a hash,
    it will error.

    When given a set of hashes, it will return their IDs. If any don't have IDs, it
    will create one and return it as part of the set.

    Args:
        start: The first integer to use for new IDs
        lookup (optional): A lookup table to use for existing hashes
    """

    def __init__(self, start: int | None = None, lookup: pa.Table | None = None):
        """Initialise the HashIDMap object."""
        self.next_int = start
        if not lookup:
            self.lookup = pa.Table.from_arrays(
                [
                    pa.array([], type=pa.uint64()),
                    pa.array([], type=pa.large_binary()),
                    pa.array([], type=pa.bool_()),
                ],
                names=["id", "hash", "new"],
            )
        else:
            new_column = pa.array([False] * lookup.shape[0], type=pa.bool_())
            self.lookup = pa.Table.from_arrays(
                [lookup["id"], lookup["hash"], new_column], names=["id", "hash", "new"]
            )

    def get_hashes(self, ids: pa.UInt64Array) -> pa.LargeBinaryArray:
        """Returns the hashes of the given IDs."""
        indices = pc.index_in(ids, self.lookup["id"])

        if pc.any(pc.is_null(indices)).as_py():
            m_mask = pc.is_null(indices)
            m_ids = pc.filter(ids, m_mask)

            raise ValueError(
                f"The following IDs were not found in lookup table: {m_ids.to_pylist()}"
            )

        return pc.take(self.lookup["hash"], indices)

    def generate_ids(self, hashes: pa.BinaryArray) -> pa.UInt64Array:
        """Returns the IDs of the given hashes, assigning new IDs for unknown hashes."""
        if self.next_int is None:
            raise RuntimeError("`next_int` was unset for HasIDMap")

        indices = pc.index_in(hashes, self.lookup["hash"])
        new_hashes = pc.unique(pc.filter(hashes, pc.is_null(indices)))

        if len(new_hashes) > 0:
            new_ids = pa.array(
                range(self.next_int, self.next_int + len(new_hashes)),
                type=pa.uint64(),
            )

            new_entries = pa.Table.from_arrays(
                [
                    new_ids,
                    new_hashes,
                    pa.array([True] * len(new_hashes), type=pa.bool_()),
                ],
                names=["id", "hash", "new"],
            )

            self.next_int += len(new_hashes)
            self.lookup = pa.concat_tables([self.lookup, new_entries])

            indices = pc.index_in(hashes, self.lookup["hash"])

        return pc.take(self.lookup["id"], indices)


def insert_source(
    source_config: SourceConfig, data_hashes: pa.Table, batch_size: int
) -> None:
    """Indexes a source within Matchbox."""
    log_prefix = f"Index {source_config.address.pretty}"
    content_hash = hash_arrow_table(data_hashes)

    with MBDB.get_session() as session:
        logger.info("Begin", prefix=log_prefix)

        # Check if resolution already exists
        existing_resolution = (
            session.query(Resolutions).filter_by(name=source_config.name).first()
        )

        if existing_resolution:
            resolution = existing_resolution
            # Check if the content hash is the same
            if resolution.hash == content_hash:
                logger.info("Source data matches index. Finished", prefix=log_prefix)
                return
        else:
            # Create new resolution
            resolution = Resolutions(
                name=source_config.name,
                hash=content_hash,
                type=ResolutionNodeType.SOURCE.value,
            )
            session.add(resolution)
            session.flush()

        # Store resolution ID for later use
        resolution_id: int = resolution.resolution_id

        # Check if source already exists
        existing_source = (
            session.query(SourceConfigs)
            .filter_by(resolution_id=resolution.resolution_id)
            .first()
        )

        if existing_source:
            logger.info("Deleting existing", prefix=log_prefix)
            session.delete(existing_source)
            session.flush()

        # Create new source with relationship to resolution
        source_obj = SourceConfigs.from_dto(resolution, source_config)
        session.add(source_obj)
        session.commit()

        logger.info(
            "Added to Resolutions, SourceConfigs, SourceFields", prefix=log_prefix
        )

        # Store source_config_id and max primary keys for later use
        source_config_id = source_obj.source_config_id

    # Don't insert new hashes, but new keys need existing hash IDs
    with MBDB.get_adbc_connection() as conn:
        existing_hash_lookup: pl.DataFrame = sql_to_df(
            stmt=compile_sql(select(Clusters.cluster_id, Clusters.cluster_hash)),
            connection=conn,
            return_type="polars",
        )

    data_hashes: pl.DataFrame = pl.from_arrow(data_hashes)
    if existing_hash_lookup.is_empty():
        new_hashes = data_hashes.select("hash").unique()
    else:
        new_hashes = (
            data_hashes.select("hash")
            .unique()
            .join(
                other=existing_hash_lookup,
                left_on="hash",
                right_on="cluster_hash",
                how="anti",
            )
        )

    if new_hashes.shape[0]:
        # Create new cluster records with sequential IDs
        next_cluster_id = PKSpace.reserve_block("clusters", len(new_hashes))
    else:
        # The value of next_cluster_id is irrelevant as cluster_records will be empty
        next_cluster_id = 0
    cluster_records = (
        new_hashes.with_row_index("cluster_id")
        .with_columns(
            [
                (pl.col("cluster_id") + next_cluster_id)
                .cast(pl.Int64)
                .alias("cluster_id")
            ]
        )
        .rename({"hash": "cluster_hash"})
    )

    # Create a combined lookup with both existing and new mappings
    lookup = pl.concat([existing_hash_lookup, cluster_records], how="vertical")

    # Add cluster_id values to data hashes
    hashes_with_ids = data_hashes.join(lookup, left_on="hash", right_on="cluster_hash")

    # Explode keys
    keys_records = (
        hashes_with_ids.select(["cluster_id", "keys"])
        .explode("keys")
        .rename({"keys": "key"})
    )

    if keys_records.shape[0] > 0:
        next_key_id = PKSpace.reserve_block("cluster_keys", len(keys_records))
    else:
        # The next_key_id is irrelevant if we don't write any keys records
        next_key_id = 0

    # Add required columns
    keys_records = keys_records.with_row_index("key_id").with_columns(
        [
            (pl.col("key_id") + next_key_id).alias("key_id"),
            pl.lit(source_config_id, dtype=pl.Int64).alias("source_config_id"),
        ]
    )

    # Insert new clusters and all source primary keys
    try:
        # Bulk insert into Clusters table (only new clusters)
        if not cluster_records.is_empty():
            large_ingest(
                data=cluster_records.to_arrow(),
                table_class=Clusters,
                max_chunksize=batch_size,
            )
            logger.info(
                f"Added {len(cluster_records):,} objects to Clusters table",
                prefix=log_prefix,
            )

        # Bulk insert into ClusterSourceKey table (all links)
        if not keys_records.is_empty():
            large_ingest(
                data=keys_records.to_arrow(),
                table_class=ClusterSourceKey,
                max_chunksize=batch_size,
            )
            logger.info(
                f"Added {len(keys_records):,} primary keys to ClusterSourceKey table",
                prefix=log_prefix,
            )
    except IntegrityError as e:
        # Log the error and rollback
        logger.warning(f"Error, rolling back: {e}", prefix=log_prefix)
        conn.rollback()

    # Insert successful, safe to update the resolution's content hash
    with MBDB.get_session() as session:
        if not keys_records.is_empty():
            stmt = (
                update(Resolutions)
                .where(Resolutions.resolution_id == resolution_id)
                .values(hash=content_hash)
            )
            session.execute(stmt)

    if cluster_records.is_empty() and keys_records.is_empty():
        logger.info("No new records to add", prefix=log_prefix)

    logger.info("Finished", prefix=log_prefix)


def insert_model(
    name: ModelResolutionName,
    left: Resolutions,
    right: Resolutions,
    description: str,
) -> None:
    """Writes a model to Matchbox with a default truth value of 100.

    Args:
        name: Name of the new model
        left: Left parent of the model
        right: Right parent of the model. Same as left in a dedupe job
        description: Model description

    Raises:
        MatchboxResolutionNotFoundError: If the specified parent models don't exist.
        MatchboxResolutionAlreadyExists: If the specified model already exists.
    """
    log_prefix = f"Model {name}"
    logger.info("Registering", prefix=log_prefix)
    with MBDB.get_session() as session:
        # Check if resolution exists
        exists_stmt = select(Resolutions).where(Resolutions.name == name)
        exists_obj = session.scalar(exists_stmt)

        if exists_obj is not None:
            raise MatchboxResolutionAlreadyExists
        else:
            new_res = Resolutions(
                type=ResolutionNodeType.MODEL.value,
                name=name,
                description=description,
                truth=100,
            )
            session.add(new_res)
            session.flush()

            def _create_closure_entries(parent_resolution: Resolutions) -> None:
                """Create closure entries for the new model.

                This is made up of mappings between nodes and any of their direct or
                indirect parents.
                """
                session.add(
                    ResolutionFrom(
                        parent=parent_resolution.resolution_id,
                        child=new_res.resolution_id,
                        level=1,
                        truth_cache=parent_resolution.truth,
                    )
                )

                ancestor_entries = (
                    session.query(ResolutionFrom)
                    .filter(ResolutionFrom.child == parent_resolution.resolution_id)
                    .all()
                )

                for entry in ancestor_entries:
                    session.add(
                        ResolutionFrom(
                            parent=entry.parent,
                            child=new_res.resolution_id,
                            level=entry.level + 1,
                            truth_cache=entry.truth_cache,
                        )
                    )

            # Create resolution lineage entries
            _create_closure_entries(parent_resolution=left)

            if right != left:
                _create_closure_entries(parent_resolution=right)
            status = "Inserted new"
            resolution_id = new_res.resolution_id

        session.commit()

    logger.info(f"{status} model with ID {resolution_id}", prefix=log_prefix)
    logger.info("Done!", prefix=log_prefix)


def _get_clusters_with_leaves(
    resolution: Resolutions,
) -> dict[int, dict[str, list[dict]]]:
    """Query clusters and their leaves, returning a nested dictionary structure.

    Args:
        resolution: Resolution object whose parents proposed the clusters
            we need to recover

    Returns:
        Dict mapping cluster_id to a dict with cluster info and leaves list
    """
    # Get parent IDs for the given resolution
    resolution_parent_ids = (
        select(Resolutions.resolution_id)
        .join(ResolutionFrom, Resolutions.resolution_id == ResolutionFrom.parent)
        .where(ResolutionFrom.child == resolution.resolution_id)
        .where(ResolutionFrom.level == 1)
        .scalar_subquery()
    )

    # Create table aliases
    ClusterRoot = alias(Clusters, "root_clusters")
    ClusterLeaf = alias(Clusters, "leaf_clusters")

    # Statement to get all cluster-leaf relationships for the resolution's parents
    stmt = (
        select(
            Contains.root.label("root_id"),
            ClusterRoot.c.cluster_hash.label("root_hash"),
            Contains.leaf.label("leaf_id"),
            ClusterLeaf.c.cluster_hash.label("leaf_hash"),
        )
        .join(ClusterRoot, Contains.root == ClusterRoot.c.cluster_id)
        .join(ClusterLeaf, Contains.leaf == ClusterLeaf.c.cluster_id)
        # Filter by resolution parent IDs
        .join(Probabilities, ClusterRoot.c.cluster_id == Probabilities.cluster)
        .join(ClusterSourceKey, ClusterLeaf.c.cluster_id == ClusterSourceKey.cluster_id)
        .join(
            SourceConfigs,
            SourceConfigs.source_config_id == ClusterSourceKey.source_config_id,
        )
        .where(
            or_(
                SourceConfigs.resolution_id.in_(resolution_parent_ids),
                Probabilities.resolution.in_(resolution_parent_ids),
            )
        )
    )

    # Create nested structure directly from query results
    result = defaultdict(lambda: {"root_hash": None, "leaves": []})  # Updated key name

    with MBDB.get_session() as session:
        for row in session.execute(stmt).yield_per(1000):
            root_id = row.root_id

            # Set cluster info if not set yet
            if result[root_id]["root_hash"] is None:
                result[root_id]["root_hash"] = row.root_hash

            # Add leaf to the leaves list
            result[root_id]["leaves"].append(
                {"leaf_id": row.leaf_id, "leaf_hash": row.leaf_hash}
            )

    return dict(result)


def _build_cluster_objects(
    nested_dict: dict[int, dict[str, list[dict]]],
) -> dict[int, Cluster]:
    """Convert the nested dictionary to Cluster objects.

    Args:
        nested_dict: Dictionary from get_clusters_with_leaves()

    Returns:
        Dict mapping cluster IDs to Cluster objects
    """
    cluster_lookup: dict[int, Cluster] = {}

    # First create all Cluster objects without leaves
    for cluster_id, data in nested_dict.items():
        cluster = Cluster(id=cluster_id, hash=data["cluster_hash"])
        cluster_lookup[cluster_id] = cluster

    # Now create and attach all leaf objects
    for cluster_id, data in nested_dict.items():
        cluster: Cluster = cluster_lookup[cluster_id]
        leaves: list[Cluster] = []

        for leaf_data in data["leaves"]:
            leaf_id = leaf_data["leaf_id"]

            # Create leaf if it doesn't exist in lookup
            if leaf_id not in cluster_lookup:
                leaf = Cluster(id=leaf_id, hash=leaf_data["leaf_hash"])
                cluster_lookup[leaf_id] = leaf
            else:
                leaf = cluster_lookup[leaf_id]

            # Add leaf to cluster's leaves list
            leaves.append(leaf)

        cluster.leaves = tuple(leaves)
        cluster_lookup[cluster_id] = cluster

    return cluster_lookup


def _results_to_cluster_pairs(
    cluster_lookup: dict[int, Cluster],
    results: pa.Table,
) -> Iterator[tuple[Cluster, Cluster, int]]:
    """Convert the results from a PyArrow table to an iterator of cluster pairs.

    Args:
        cluster_lookup (dict[int, Cluster]): A dictionary mapping cluster IDs to
            Cluster objects.
        results (pa.Table): The PyArrow table containing the results, left_id
            right_id, and probability.

    Returns:
        list[tuple[Cluster, Cluster, int]]: An iterator of tuples, each containing
            the left cluster, right cluster, and the probability, in descending
            order of probability.
    """
    for row in pl.from_arrow(results).sort("probability", descending=True).iter_rows():
        left_cluster: Cluster = cluster_lookup[row["left_id"]]
        right_cluster: Cluster = cluster_lookup[row["right_id"]]

        yield left_cluster, right_cluster, row["probability"]


def _results_to_insert_tables(
    resolution: Resolutions, probabilities: pa.Table
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """Takes probabilities and returns three Arrow tables that can be inserted exactly.

    Returns:
        A tuple containing:

            * A Clusters update Arrow table
            * A Contains update Arrow table
            * A Probabilities update Arrow table
    """
    log_prefix = f"Model {resolution.name}"

    if probabilities.shape[0] == 0:
        clusters = pa.table(
            {"cluster_id": [], "cluster_hash": []},
            schema=pa.schema(
                [("cluster_id", pa.uint64()), ("cluster_hash", pa.large_binary())]
            ),
        )
        contains = pa.table(
            {"root": [], "leaf": []},
            schema=pa.schema([("root", pa.uint64()), ("leaf", pa.uint64())]),
        )
        probabilities = pa.table(
            {"resolution": [], "cluster": [], "probability": []},
            schema=pa.schema(
                [
                    ("resolution", pa.uint64()),
                    ("cluster", pa.uint64()),
                    ("probability", pa.uint8()),
                ]
            ),
        )
        return clusters, contains, probabilities

    logger.info("Wrangling data to insert tables", prefix=log_prefix)

    # Get a cluster lookup dictionary based on the resolution's parents
    nested_data = _get_clusters_with_leaves(resolution=resolution)
    cluster_lookup: dict[int, Cluster] = _build_cluster_objects(nested_data)

    # Calculate hierarchies
    logger.debug("Computing hierarchies", prefix=log_prefix)

    djs = DisjointSet[Cluster]()

    new_clusters: dict[bytes, Clusters] = {}
    probabilities_dict: dict[bytes, int] = {}
    threshold: int = 100
    for left_cluster, right_cluster, probability in _results_to_cluster_pairs(
        cluster_lookup, probabilities
    ):
        # Process pairwise probabilities
        pair_cluster = Cluster.combine_many(left_cluster, right_cluster)
        new_clusters[pair_cluster.hash] = pair_cluster
        probabilities_dict[pair_cluster.hash] = probability

        if probability < threshold:
            # Process the components at the previous threshold
            components: list[set[Cluster]] = djs.get_components()
            for component in components:
                cluster = Cluster.combine_many(component)
                new_clusters[cluster.hash] = cluster
                probabilities_dict[cluster.hash] = probability

            # Continue to next threshold
            threshold = probability

        djs.union(left_cluster, right_cluster)

    new_cluster_table = pa.Table.from_arrays(
        [pa.array(list(new_clusters.keys()), pa.large_binary())], names=["cluster_hash"]
    )
    with ingest_to_temporary_table(
        table_name="tmp_hashes", schema_name="mb", data=new_cluster_table
    ) as temp_table:
        existing_cluster_stmt = select(
            Clusters.cluster_id, Clusters.cluster_hash()
        ).join(temp_table, temp_table.c.cluster_hash == Cluster.cluster_hash)

        with MBDB.get_adbc_connection() as conn:
            cluster_hash_lookup: pl.DataFrame = sql_to_df(
                stmt=compile_sql(existing_cluster_stmt),
                connection=conn,
                return_type="polars",
            )

    # Create a polars DataFrame with all hashes from new_clusters
    all_clusters_df = pl.DataFrame({"cluster_hash": list(new_clusters.keys())})

    logger.debug("Reconciling new hashes against the db", prefix=log_prefix)

    # Use anti_join to find hashes that don't exist in the lookup
    missing_clusters_df = all_clusters_df.join(
        cluster_hash_lookup, on="cluster_hash", how="anti"
    )

    if missing_clusters_df.shape[0]:
        # Create new cluster records with sequential IDs
        next_cluster_id = PKSpace.reserve_block(
            "clusters", missing_clusters_df.shape[0]
        )

        # Add cluster_id column with sequential values starting from next_cluster_id
        missing_clusters_df = missing_clusters_df.with_columns(
            pl.arange(0, missing_clusters_df.shape[0], dtype=pl.Int64)
            .add(next_cluster_id)
            .alias("cluster_id")
        )
    else:
        missing_clusters_df = missing_clusters_df.with_columns(
            pl.lit(None).cast(pl.Int64).alias("cluster_id")
        )

    missing_clusters_df = missing_clusters_df.select(["cluster_id", "cluster_hash"])

    # Concatenate with existing lookup to get the complete lookup table
    cluster_hash_lookup = pl.concat([cluster_hash_lookup, missing_clusters_df])

    contains_list: list[dict[str, int | list[int]]] = []
    for row in missing_clusters_df.iter_rows():
        # Access cluster_hash, index 1 for tuple
        cluster = new_clusters[row[1]]
        contains_list.append(
            {"root": row[0], "leaf": [leaf.id for leaf in cluster.leaves]}
        )

    contains_df = pl.DataFrame(contains_list).explode("leaf")

    probabilities_df = (
        pl.from_dict(
            {
                "cluster_hash": probabilities_dict.keys(),
                "probability": probabilities_dict.values(),
            }
        )
        .join(cluster_hash_lookup, on="cluster_hash", how="left")
        .select([pl.col("cluster_id").alias("cluster"), "probability"])
        .with_columns(
            pl.lit(resolution.resolution_id).cast(pl.Int64).alias("resolution")
        )
        .select(["resolution", "cluster", "probability"])
    )

    logger.info("Wrangling complete!", prefix=log_prefix)

    return (
        missing_clusters_df.to_arrow(),
        contains_df.to_arrow(),
        probabilities_df.to_arrow(),
    )


def insert_results(
    resolution: Resolutions,
    results: pa.Table,
    batch_size: int,
) -> None:
    """Writes a results table to Matchbox.

    The PostgreSQL backend stores clusters in a hierarchical structure, where
    each component references its parent component at a higher threshold.

    This means two-item components are synonymous with their original pairwise
    probabilities.

    This allows easy querying of clusters at any threshold.

    Args:
        resolution: Resolution of type model to associate results with
        results: A PyArrow results table with left_id, right_id, probability
        batch_size: Number of records to insert in each batch

    Raises:
        MatchboxResolutionNotFoundError: If the specified model doesn't exist.
    """
    log_prefix = f"Model {resolution.name}"
    logger.info(
        f"Writing results data with batch size {batch_size:,}", prefix=log_prefix
    )

    # Check if the content hash is the same
    content_hash = hash_arrow_table(results)
    if resolution.hash == content_hash:
        logger.info("Results already uploaded. Finished", prefix=log_prefix)
        return

    clusters, contains, probabilities = _results_to_insert_tables(
        resolution=resolution, probabilities=results
    )

    with MBDB.get_session() as session:
        try:
            # Clear existing probabilities for this resolution
            stmt = delete(Probabilities).where(
                Probabilities.resolution == resolution.resolution_id
            )
            session.execute(stmt)

            session.commit()
            logger.info("Removed old probabilities", prefix=log_prefix)

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(
                f"Failed to clear old probabilities or update content hash: {str(e)}",
                prefix=log_prefix,
            )
            raise

    try:
        logger.info(
            f"Inserting {clusters.shape[0]:,} results objects", prefix=log_prefix
        )

        large_ingest(
            data=clusters,
            table_class=Clusters,
            max_chunksize=batch_size,
        )

        logger.info(
            f"Successfully inserted {clusters.shape[0]:,} objects into Clusters table",
            prefix=log_prefix,
        )

        large_ingest(
            data=contains,
            table_class=Contains,
            max_chunksize=batch_size,
        )

        logger.info(
            f"Successfully inserted {contains.shape[0]:,} objects into Contains table",
            prefix=log_prefix,
        )

        large_ingest(
            data=probabilities,
            table_class=Probabilities,
            max_chunksize=batch_size,
        )

        logger.info(
            f"Successfully inserted "
            f"{probabilities.shape[0]:,} objects into Probabilities table",
            prefix=log_prefix,
        )

    except SQLAlchemyError as e:
        logger.error(f"Failed to insert data: {str(e)}", prefix=log_prefix)
        raise

    # Insert successful, safe to update the resolution's content hash
    with MBDB.get_session() as session:
        stmt = (
            update(Resolutions)
            .where(Resolutions.resolution_id == resolution.resolution_id)
            .values(hash=content_hash)
        )
        session.execute(stmt)
        session.commit()

    logger.info("Insert operation complete!", prefix=log_prefix)
