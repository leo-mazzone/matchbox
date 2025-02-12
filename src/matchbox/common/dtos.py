from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from matchbox.common.arrow import SCHEMA_INDEX, SCHEMA_RESULTS


class BackendCountableType(StrEnum):
    DATASETS = "datasets"
    MODELS = "models"
    DATA = "data"
    CLUSTERS = "clusters"
    CREATES = "creates"
    MERGES = "merges"
    PROPOSES = "proposes"


class ModelResultsType(StrEnum):
    PROBABILITIES = "probabilities"
    CLUSTERS = "clusters"


class BackendRetrievableType(StrEnum):
    SOURCE = "source"
    RESOLUTION = "resolution"


class BackendUploadType(StrEnum):
    INDEX = "index"
    RESULTS = "results"

    @property
    def schema(self):
        return {
            BackendUploadType.INDEX: SCHEMA_INDEX,
            BackendUploadType.RESULTS: SCHEMA_RESULTS,
        }[self]


class ModelType(StrEnum):
    """Enumeration of supported model types."""

    LINKER = "linker"
    DEDUPER = "deduper"


class ModelOperationType(StrEnum):
    """Enumeration of supported model operations."""

    INSERT = "insert"
    UPDATE_TRUTH = "update_truth"
    UPDATE_ANCESTOR_CACHE = "update_ancestor_cache"
    DELETE = "delete"


class ModelMetadata(BaseModel):
    """Metadata for a model."""

    name: str
    description: str
    type: ModelType
    left_resolution: str
    right_resolution: str | None = None  # Only used for linker models


class ModelAncestor(BaseModel):
    """A model's ancestor and its truth value."""

    name: str = Field(..., description="Name of the ancestor model")
    truth: float | None = Field(
        default=None, description="Truth threshold value", ge=0.0, le=1.0
    )


class ModelOperationStatus(BaseModel):
    """Status response for any model operation."""

    success: bool
    model_name: str
    operation: ModelOperationType
    details: str | None = None

    @classmethod
    def status_409_examples(cls) -> dict:
        return {
            "content": {
                "application/json": {
                    "examples": {
                        "confirm_delete": {
                            "summary": "Delete operation requires confirmation. ",
                            "value": cls(
                                success=False,
                                model_name="example_model",
                                operation=ModelOperationType.DELETE,
                                details=(
                                    "This operation will delete the resolutions "
                                    "deduper_1, deduper_2, linker_1, "
                                    "as well as all probabilities they have created. "
                                    "\n\n"
                                    "It won't delete validation associated with these "
                                    "probabilities. \n\n"
                                    "If you're sure you want to continue, rerun with "
                                    "certain=True"
                                ),
                            ).model_dump(),
                        },
                    },
                }
            }
        }

    @classmethod
    def status_500_examples(cls) -> dict:
        return {
            "content": {
                "application/json": {
                    "examples": {
                        "unhandled": {
                            "summary": (
                                "Unhandled exception encountered while updating the "
                                "model's truth value."
                            ),
                            "value": cls(
                                success=False,
                                model_name="example_model",
                                operation=ModelOperationType.UPDATE_TRUTH,
                            ).model_dump(),
                        },
                    },
                }
            }
        }


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


class CountResult(BaseModel):
    """Response model for count results"""

    entities: dict[BackendCountableType, int]


class UploadStatus(BaseModel):
    """Response model for any file upload processes, like Source or Model results."""

    id: str | None = None
    status: Literal[
        "ready", "awaiting_upload", "queued", "processing", "complete", "failed"
    ]
    details: str | None = None
    entity: BackendUploadType | None = None

    @classmethod
    def status_400_examples(cls) -> dict:
        return {
            "content": {
                "application/json": {
                    "examples": {
                        "expired_id": {
                            "summary": "Upload ID expired",
                            "value": cls(
                                id="example_id",
                                status="failed",
                                details=(
                                    "Upload ID not found or expired. Entries expire "
                                    "after 30 minutes of inactivity, including "
                                    "failed processes."
                                ),
                                entity=BackendUploadType.INDEX,
                            ).model_dump(),
                        },
                        "schema_mismatch": {
                            "summary": "Schema validation error",
                            "value": cls(
                                id="example_id",
                                status="failed",
                                details="Schema mismatch. Expected: ... Got: ...",
                                entity=BackendUploadType.INDEX,
                            ).model_dump(),
                        },
                    },
                }
            }
        }


class NotFoundError(BaseModel):
    """API error for a 404 status code"""

    details: str
    entity: BackendRetrievableType
