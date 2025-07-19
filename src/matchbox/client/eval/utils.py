"""Collection of client-side functions in aid of model evaluation."""

import warnings
from typing import Any

import polars as pl
from matplotlib import pyplot as plt
from sqlalchemy import create_engine

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.results import Results
from matchbox.common.dtos import ModelResolutionName
from matchbox.common.eval import (
    ModelComparison,
    PrecisionRecall,
    precision_recall,
)
from matchbox.common.graph import DEFAULT_RESOLUTION
from matchbox.common.logging import logger


def get_samples(
    n: int,
    user_id: int,
    resolution: ModelResolutionName | None = None,
    credentials: Any | None = None,
) -> dict[int, pl.DataFrame]:
    """Retrieve samples enriched with source data, grouped by resolution cluster.

    Args:
        n: Number of clusters to sample
        user_id: ID of the user requesting the samples
        resolution: Model resolution proposing the clusters. If not set, will
            use a default resolution.
        credentials: Valid credentials for the source configs.
            Sources that can't be queried with these credentials will be skipped.
            If not provided, will populate with a SQLAlchemy engine
            from the default warehouse set in the environment variable
            `MB__CLIENT__DEFAULT_WAREHOUSE`

    Returns:
        Dictionary of cluster ID to dataframe describing the cluster
    """
    if not resolution:
        resolution = DEFAULT_RESOLUTION
    if not credentials:
        if default_credentials := settings.default_warehouse:
            credentials = create_engine(default_credentials)
            logger.warning("Using default engine")
        else:
            raise ValueError(
                "Credentials need to be provided if "
                "`MB__CLIENT__DEFAULT_WAREHOUSE` is unset"
            )

    samples: pl.DataFrame = pl.from_arrow(
        _handler.sample_for_eval(n=n, resolution=resolution, user_id=user_id)
    )

    if not len(samples):
        return {}

    results_by_source = []
    for source_resolution in samples["source"].unique():
        source_config = _handler.get_source_config(source_resolution)
        try:
            source_config.location.add_credentials(credentials=credentials)
        except ValueError:
            warnings.warn(
                f"Skipping {source_resolution}, incompatible with given credentials.",
                UserWarning,
                stacklevel=2,
            )
            continue

        samples_by_source = samples.filter(pl.col("source") == source_resolution)
        keys_by_source = samples_by_source["key"].to_list()

        source_data = pl.concat(
            source_config.query(
                batch_size=10_000, qualify_names=True, keys=keys_by_source
            )
        )
        samples_and_source = samples_by_source.join(
            source_data, left_on="key", right_on=source_config.qualified_key
        )
        desired_columns = ["root", "leaf", "key"] + source_config.qualified_fields
        results_by_source.append(samples_and_source[desired_columns])

    if not len(results_by_source):
        return {}

    all_results: pl.DataFrame = pl.concat(results_by_source, how="diagonal")

    results_by_root = {
        root: all_results.filter(pl.col("root") == root).drop("root")
        for root in all_results["root"].unique()
    }

    return results_by_root


class EvalData:
    """Object which caches evaluation data to measure performance of models."""

    # TODO: test this object
    def __init__(self):
        """Initialise evaluation data from resolution name."""
        self.judgements, self.expansion = _handler.download_eval_data()

    def precision_recall(self, results: Results, threshold: float) -> PrecisionRecall:
        """Computes precision and recall at one threshold."""
        threshold = int(threshold * 100)

        root_leaf = (
            results.root_leaf()
            .rename({"root_id": "root", "leaf_id": "leaf"})
            .to_arrow()
        )
        return precision_recall([root_leaf], self.judgements, self.expansion)[0]

    def pr_curve(self, results: Results) -> dict[str, PrecisionRecall]:
        """For each threshold in retults computes precision and recall."""
        all_p = []
        all_r = []

        probs = pl.from_arrow(results.probabilities)
        thresholds = probs.select("probability").unique().to_series()
        for i, t in enumerate(sorted(thresholds)):
            float_thresh = t / 100
            p, r = self.precision_recall(results=results, threshold=float_thresh)
            all_p.append(p)
            all_r.append(r)
            plt.annotate(float_thresh, (all_r[i], all_p[i]))

        plt.plot(all_r, all_p, marker="o")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title("Precision-Recall Curve")
        plt.grid()

        return plt


def compare_models(resolutions: list[ModelResolutionName]) -> ModelComparison:
    """Compare metrics of models based on evaluation data.

    Args:
        resolutions: List of names of model resolutions to be compared.

    Returns:
        A model comparison object, listing metrics for each model.
    """
    return _handler.compare_models(resolutions)
