import inspect
from abc import ABC, abstractmethod
from enum import StrEnum
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
)

import boto3
from botocore.exceptions import ClientError
from dotenv import find_dotenv, load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine

from matchbox.common.db import Match, Source
from matchbox.common.graph import ResolutionGraph

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
    from pandas import DataFrame as PandasDataFrame
    from polars import DataFrame as PolarsDataFrame
    from pyarrow import Table as ArrowTable

    from matchbox.client.results import Results
else:
    S3Client = Any
    PandasDataFrame = Any
    PolarsDataFrame = Any
    ArrowTable = Any

    Results = Any


dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path, override=True)

R = TypeVar("R")
P = ParamSpec("P")


class MatchboxBackends(StrEnum):
    """The available backends for Matchbox."""

    POSTGRES = "postgres"


class MatchboxDatastoreSettings(BaseSettings):
    """Settings specific to the datastore configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MB__DATASTORE__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str
    port: str
    access_key_id: SecretStr
    secret_access_key: SecretStr
    default_region: str
    cache_bucket_name: str

    def get_client(self) -> S3Client:
        """Returns an S3 client for the datastore.

        Creates S3 buckets if they don't exist.
        """
        client = boto3.client(
            "s3",
            endpoint_url=f"http://{self.host}:{self.port}",
            aws_access_key_id=self.access_key_id.get_secret_value(),
            aws_secret_access_key=self.secret_access_key.get_secret_value(),
            region_name=self.default_region,
        )

        try:
            client.head_bucket(Bucket=self.cache_bucket_name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                client.create_bucket(
                    Bucket=self.cache_bucket_name,
                    CreateBucketConfiguration={
                        "LocationConstraint": self.default_region
                    },
                )
            else:
                raise e


class MatchboxSettings(BaseSettings):
    """Settings for the Matchbox application."""

    model_config = SettingsConfigDict(
        env_prefix="MB__",
        env_nested_delimiter="__",
        use_enum_values=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    datasets_config: Path
    batch_size: int = Field(default=250_000)
    backend_type: MatchboxBackends
    datastore: MatchboxDatastoreSettings


class BackendManager:
    """Manages the Matchbox backend instance and settings."""

    _instance = None
    _settings = None

    @classmethod
    def initialise(cls, settings: "MatchboxSettings"):
        cls._settings = settings

    @classmethod
    def get_backend(cls) -> "MatchboxDBAdapter":
        if cls._settings is None:
            raise ValueError("BackendManager must be initialized with settings first")

        if cls._instance is None:
            BackendClass = get_backend_class(cls._settings.backend_type)
            cls._instance = BackendClass(cls._settings)
        return cls._instance

    @classmethod
    def get_settings(cls) -> "MatchboxSettings":
        if cls._settings is None:
            raise ValueError("BackendManager must be initialized with settings first")
        return cls._settings


def get_backend_settings(backend_type: MatchboxBackends) -> type[MatchboxSettings]:
    """Get the appropriate settings class based on the backend type."""
    if backend_type == MatchboxBackends.POSTGRES:
        from matchbox.server.postgresql import MatchboxPostgresSettings

        return MatchboxPostgresSettings
    # Add more backend types here as needed
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def get_backend_class(backend_type: MatchboxBackends) -> type["MatchboxDBAdapter"]:
    """Get the appropriate backend class based on the backend type."""
    if backend_type == MatchboxBackends.POSTGRES:
        from matchbox.server.postgresql import MatchboxPostgres

        return MatchboxPostgres
    # Add more backend types here as needed
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def initialise_backend(settings: MatchboxSettings) -> None:
    """Utility function to initialise the Matchbox backend based on settings."""
    BackendManager.initialise(settings)


def initialise_matchbox() -> None:
    """Initialise the Matchbox backend based on environment variables."""
    base_settings = MatchboxSettings()

    SettingsClass = get_backend_settings(base_settings.backend_type)
    settings = SettingsClass()

    initialise_backend(settings)


def inject_backend(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator to inject the Matchbox backend into functions.

    Used to allow user-facing functions to access the backend without needing to
    pass it in. The backend is defined by the MB__BACKEND_TYPE environment variable.

    Can be used for both functions and methods.

    If the user specifies a backend, it will be used instead of the injection.
    """

    @wraps(func)
    def _inject_backend(
        *args: P.args, backend: "MatchboxDBAdapter | None" = None, **kwargs: P.kwargs
    ) -> R:
        if backend is None:
            backend = BackendManager.get_backend()

        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if params and params[0].name in ("self", "cls"):
            return cast(R, func(args[0], backend, *args[1:], **kwargs))
        else:
            return cast(R, func(backend, *args, **kwargs))

    return cast(Callable[..., R], _inject_backend)


class Countable(Protocol):
    """A protocol for objects that can be counted."""

    def count(self) -> int: ...


class Listable(Protocol):
    """A protocol for objects that can be listed."""

    def list(self) -> list[str]: ...


class ListableAndCountable(Countable, Listable):
    """A protocol for objects that can be counted and listed."""

    pass


class MatchboxModelAdapter(ABC):
    """An abstract base class for Matchbox model adapters.

    Must be able to recover probabilities and clusters from the database,
    but ultimately doesn't care how they're stored.

    Creates these with the pairwise probabilities and the connected components
    of those pairs calculated at every threshold.
    """

    id: int
    hash: bytes
    name: str

    @property
    @abstractmethod
    def results(self) -> Results: ...

    @results.setter
    @abstractmethod
    def results(self, results: Results) -> None: ...

    @property
    @abstractmethod
    def truth(self) -> float: ...

    @truth.setter
    @abstractmethod
    def truth(self, truth: float) -> None: ...

    @property
    @abstractmethod
    def ancestors(self) -> dict[str, float]: ...

    @property
    @abstractmethod
    def ancestors_cache(self) -> dict[str, float]: ...

    @ancestors_cache.setter
    @abstractmethod
    def ancestors_cache(self, ancestors_cache: dict[str, float]) -> None: ...


class MatchboxDBAdapter(ABC):
    """An abstract base class for Matchbox database adapters."""

    settings: "MatchboxSettings"

    datasets: ListableAndCountable
    models: Countable
    data: Countable
    clusters: Countable
    creates: Countable
    merges: Countable
    proposes: Countable

    @abstractmethod
    def query(
        self,
        selector: dict[str, list[str]],
        resolution: str | None = None,
        threshold: float | dict[str, float] | None = None,
        return_type: Literal["pandas", "arrow", "polars"] | None = None,
        limit: int = None,
    ) -> PandasDataFrame | ArrowTable | PolarsDataFrame: ...

    @abstractmethod
    def match(
        self,
        source_pk: str,
        source: str,
        target: str | list[str],
        resolution: str,
        threshold: float | dict[str, float] | None = None,
    ) -> Match | list[Match]: ...

    @abstractmethod
    def index(self, dataset: Source) -> None: ...

    @abstractmethod
    def validate_ids(self, ids: list[int]) -> bool: ...

    @abstractmethod
    def validate_hashes(self, hashes: list[bytes]) -> bool: ...

    @abstractmethod
    def cluster_id_to_hash(self, ids: list[int]) -> dict[int, bytes | None]: ...

    @abstractmethod
    def get_dataset(self, db_schema: str, db_table: str, engine: Engine) -> Source: ...

    @abstractmethod
    def get_resolution_graph(self) -> ResolutionGraph: ...

    @abstractmethod
    def get_model(self, model: str) -> MatchboxModelAdapter: ...

    @abstractmethod
    def delete_model(self, model: str, certain: bool) -> None: ...

    @abstractmethod
    def insert_model(self, model: str) -> None: ...

    @abstractmethod
    def clear(self, certain: bool) -> None: ...
