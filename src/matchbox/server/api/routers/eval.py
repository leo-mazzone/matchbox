"""Resolutions API routes for the Matchbox server."""

from fastapi import APIRouter

from matchbox.common.dtos import (
    ModelResolutionName,
    ResolutionOperationStatus,
)
from matchbox.common.eval import ModelComparison
from matchbox.server.api.dependencies import (
    BackendDependency,
)

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post(
    "/login",
)
async def login(
    backend: BackendDependency,
    user_id: int,
) -> str:
    """Receives an identity and returns user information."""


@router.post(
    "/",
)
async def insert_judgement(
    backend: BackendDependency,
    resolutions: list[ModelResolutionName],
):
    """Submit judgement from human evaluator."""


@router.get(
    "/compare",
)
async def compare_models(
    backend: BackendDependency,
    resolutions: list[ModelResolutionName],
) -> ModelComparison:
    """Return comparison of selected models."""
    return {
        "companieshouse_datahub_deterministic": (1, 0.8),
        "companieshouse_datahub_probabilistic": (0.8, 0.9),
    }


@router.get(
    "/sample",
)
async def sample_one(
    backend: BackendDependency,
    resolution: ModelResolutionName,
) -> ResolutionOperationStatus:
    """Sample a cluster to validate."""
