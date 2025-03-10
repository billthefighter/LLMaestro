import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiofiles
from pydantic import ConfigDict, Field
from llmaestro.core.persistence import PersistentModel
from sqlalchemy import JSON, Column, DateTime, String, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base

Base = declarative_base()


class ArtifactModel(Base):
    """SQLAlchemy model for storing artifacts."""

    __tablename__ = "artifacts"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    data = Column(JSON, nullable=False)
    path = Column(String, nullable=True)
    timestamp = Column(DateTime, nullable=False)
    artifact_metadata = Column(JSON, nullable=False)

    @classmethod
    def from_pydantic(cls, artifact: "Artifact") -> "ArtifactModel":
        """Create a database record from a pydantic Artifact."""
        return cls(
            id=artifact.id,
            name=artifact.name,
            content_type=artifact.content_type,
            data=artifact.serialize(),
            path=str(artifact.path) if artifact.path else None,
            timestamp=artifact.timestamp,
            artifact_metadata=artifact.metadata,
        )

    def to_pydantic(self) -> "Artifact":
        """Convert database record to a pydantic Artifact."""
        # Get actual values from SQLAlchemy columns
        path_str = str(self.path) if self.path is not None else None
        path = Path(path_str) if path_str else None

        # Handle timestamp
        try:
            timestamp = self.timestamp.replace(tzinfo=None)
        except (AttributeError, TypeError):
            timestamp = datetime.now()

        return Artifact(
            id=str(self.id),
            name=str(self.name),
            content_type=str(self.content_type),
            data=dict(self.data) if isinstance(self.data, dict) else self.data,
            path=path,
            timestamp=timestamp,
            metadata=dict(self.artifact_metadata) if isinstance(self.artifact_metadata, dict) else {},
        )


class Artifact(PersistentModel):
    """Base model for all artifacts in the system."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    content_type: str
    data: Any
    path: Optional[Path] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def serialize(self) -> Any:
        """Serialize the artifact data for storage.

        Returns:
            The serialized data in a format suitable for storage (dict, list, or primitive type).
        """
        if hasattr(self.data, "model_dump"):
            return self.data.model_dump()
        elif isinstance(self.data, list) and all(hasattr(item, "model_dump") for item in self.data):
            return [item.model_dump() for item in self.data]
        return self.data

    def to_orm(self) -> ArtifactModel:
        """Convert to SQLAlchemy ORM model."""
        return ArtifactModel.from_pydantic(self)


class ArtifactStorage(PersistentModel):
    """Base class defining the interface for artifact storage implementations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def save_artifact(self, artifact: Artifact) -> bool:
        """Save an artifact to storage."""
        raise NotImplementedError

    def load_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Load an artifact from storage."""
        raise NotImplementedError

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact from storage."""
        raise NotImplementedError

    def list_artifacts(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Artifact]:
        """List artifacts matching the filter criteria."""
        raise NotImplementedError


class DatabaseArtifactStorage(ArtifactStorage):
    """Database implementation of artifact storage using SQLAlchemy."""

    def __init__(self, connection_string: str):
        """Initialize database storage.

        Args:
            connection_string: SQLAlchemy connection string (e.g., 'sqlite:///artifacts.db')
        """
        super().__init__()
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)

    def save_artifact(self, artifact: Artifact) -> bool:
        """Save an artifact to the database."""
        try:
            with Session(self.engine) as session:
                db_artifact = artifact.to_orm()
                session.merge(db_artifact)  # Use merge instead of add to handle updates
                session.commit()
            return True
        except SQLAlchemyError as e:
            print(f"Error saving artifact to database: {e}")
            return False

    def load_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Load an artifact from the database."""
        try:
            with Session(self.engine) as session:
                db_artifact = session.query(ArtifactModel).get(artifact_id)
                if not db_artifact:
                    return None
                return db_artifact.to_pydantic()
        except SQLAlchemyError as e:
            print(f"Error loading artifact from database: {e}")
            return None

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact from the database."""
        try:
            with Session(self.engine) as session:
                db_artifact = session.query(ArtifactModel).get(artifact_id)
                if db_artifact:
                    session.delete(db_artifact)
                    session.commit()
                    return True
                return False
        except SQLAlchemyError as e:
            print(f"Error deleting artifact from database: {e}")
            return False

    def list_artifacts(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Artifact]:
        """List artifacts matching the filter criteria."""
        try:
            with Session(self.engine) as session:
                query = session.query(ArtifactModel)

                # Apply filters if provided
                if filter_criteria:
                    for key, value in filter_criteria.items():
                        if hasattr(ArtifactModel, key):
                            query = query.filter(getattr(ArtifactModel, key) == value)

                return [db_artifact.to_pydantic() for db_artifact in query.all()]
        except SQLAlchemyError as e:
            print(f"Error listing artifacts from database: {e}")
            return []


class StorageConfig(PersistentModel):
    """Configuration for artifact storage. This is pretty barebones right now, but will be expanded in the future."""

    base_path: Path
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FileSystemArtifactStorage(ArtifactStorage):
    """File system implementation of artifact storage."""

    config: StorageConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(cls, base_path: Path) -> "FileSystemArtifactStorage":
        """Create a new FileSystemArtifactStorage instance."""
        config = StorageConfig(base_path=base_path)
        instance = cls(config=config)
        instance._ensure_storage_path()
        return instance

    def __init__(self, config: StorageConfig):
        super().__init__(config=config)
        self._ensure_storage_path()

    def _ensure_storage_path(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.config.base_path.mkdir(parents=True, exist_ok=True)

    def _get_artifact_path(self, artifact_id: str) -> Path:
        """Get the file path for an artifact."""
        return self.config.base_path / f"{artifact_id}.json"

    def save_artifact(self, artifact: Artifact) -> bool:
        """Save an artifact to the file system."""
        try:
            artifact_path = self._get_artifact_path(artifact.id)

            # Update artifact path
            artifact.path = artifact_path

            # Serialize and save
            data = {
                "id": artifact.id,
                "name": artifact.name,
                "content_type": artifact.content_type,
                "data": artifact.serialize(),
                "path": str(artifact.path),
                "timestamp": artifact.timestamp.isoformat(),
                "metadata": artifact.metadata,
            }

            with open(artifact_path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving artifact {artifact.id}: {e}")
            return False

    def load_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Load an artifact from the file system."""
        artifact_path = self._get_artifact_path(artifact_id)
        if not artifact_path.exists():
            return None

        try:
            with open(artifact_path) as f:
                data = json.load(f)
                # Convert path back to Path object
                data["path"] = Path(data["path"]) if data.get("path") else None
                # Parse timestamp
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                return Artifact(**data)
        except Exception as e:
            print(f"Error loading artifact {artifact_id}: {e}")
            return None

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact from the file system."""
        artifact_path = self._get_artifact_path(artifact_id)
        try:
            if artifact_path.exists():
                artifact_path.unlink()
                return True
            return False
        except Exception as e:
            print(f"Error deleting artifact {artifact_id}: {e}")
            return False

    def list_artifacts(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Artifact]:
        """List artifacts matching the filter criteria."""
        artifacts = []
        for path in self.config.base_path.glob("*.json"):
            try:
                artifact = self.load_artifact(path.stem)
                if artifact:
                    # Apply filters if provided
                    if filter_criteria:
                        matches = True
                        for key, value in filter_criteria.items():
                            if not hasattr(artifact, key) or getattr(artifact, key) != value:
                                matches = False
                                break
                        if not matches:
                            continue
                    artifacts.append(artifact)
            except Exception as e:
                print(f"Error loading artifact from {path}: {e}")
                continue
        return artifacts

    async def save_artifact_async(self, artifact: Artifact) -> bool:
        """Asynchronous artifact saving."""
        if not artifact.path:
            return False

        try:
            path_str = str(artifact.path)
            async with aiofiles.open(path_str, "w") as f:
                await f.write(json.dumps(artifact.data))
            return True
        except Exception as e:
            print(f"Async artifact save error: {e}")
            return False

    async def load_artifact_async(self, artifact_id: str) -> Optional[Artifact]:
        """Asynchronous artifact loading."""
        try:
            path_str = str(self._get_artifact_path(artifact_id))
            async with aiofiles.open(path_str, "r") as f:
                content = await f.read()
                return Artifact.parse_raw(content)
        except Exception as e:
            print(f"Async artifact load error: {e}")
            return None
