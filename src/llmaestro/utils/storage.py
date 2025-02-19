import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from sqlalchemy import JSON, Column, DateTime, String, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base

from ..core.models import Artifact, ArtifactStorage, StorageConfig

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


class DatabaseArtifactStorage(ArtifactStorage):
    """Database implementation of artifact storage using SQLAlchemy."""

    def __init__(self, connection_string: str):
        """Initialize database storage.

        Args:
            connection_string: SQLAlchemy connection string (e.g., 'sqlite:///artifacts.db')
        """
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)

    def save_artifact(self, artifact: Artifact) -> bool:
        """Save an artifact to the database."""
        try:
            with Session(self.engine) as session:
                db_artifact = ArtifactModel(
                    id=artifact.id,
                    name=artifact.name,
                    content_type=artifact.content_type,
                    data=artifact.serialize(),
                    path=str(artifact.path) if artifact.path else None,
                    timestamp=artifact.timestamp,
                    artifact_metadata=artifact.metadata,
                )
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

                # Get actual values from SQLAlchemy columns
                path_str = str(db_artifact.path) if db_artifact.path is not None else None
                path = Path(path_str) if path_str else None

                # Handle timestamp
                try:
                    timestamp = db_artifact.timestamp.replace(tzinfo=None)
                except (AttributeError, TypeError):
                    timestamp = datetime.now()

                return Artifact(
                    id=str(db_artifact.id),
                    name=str(db_artifact.name),
                    content_type=str(db_artifact.content_type),
                    data=dict(db_artifact.data) if isinstance(db_artifact.data, dict) else db_artifact.data,
                    path=path,
                    timestamp=timestamp,
                    metadata=dict(db_artifact.artifact_metadata)
                    if isinstance(db_artifact.artifact_metadata, dict)
                    else {},
                )
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

                artifacts = []
                for db_artifact in query.all():
                    # Get actual values from SQLAlchemy columns
                    path_str = str(db_artifact.path) if db_artifact.path is not None else None
                    path = Path(path_str) if path_str else None

                    # Handle timestamp
                    try:
                        timestamp = db_artifact.timestamp.replace(tzinfo=None)
                    except (AttributeError, TypeError):
                        timestamp = datetime.now()

                    artifacts.append(
                        Artifact(
                            id=str(db_artifact.id),
                            name=str(db_artifact.name),
                            content_type=str(db_artifact.content_type),
                            data=dict(db_artifact.data) if isinstance(db_artifact.data, dict) else db_artifact.data,
                            path=path,
                            timestamp=timestamp,
                            metadata=dict(db_artifact.artifact_metadata)
                            if isinstance(db_artifact.artifact_metadata, dict)
                            else {},
                        )
                    )
                return artifacts
        except SQLAlchemyError as e:
            print(f"Error listing artifacts from database: {e}")
            return []


class FileSystemArtifactStorage(ArtifactStorage):
    """File system implementation of artifact storage."""

    def __init__(self, base_path: str):
        self.config = StorageConfig(base_path=base_path)
        self.base_path = Path(base_path)
        self._ensure_storage_path()

    def _ensure_storage_path(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_artifact_path(self, artifact_id: str) -> Path:
        """Get the file path for an artifact."""
        return self.base_path / f"{artifact_id}.json"

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
        for path in self.base_path.glob("*.json"):
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
        try:
            async with aiofiles.open(artifact.path, "w") as f:
                await f.write(json.dumps(artifact.data))
            return True
        except Exception as e:
            print(f"Async artifact save error: {e}")
            return False

    async def load_artifact_async(self, artifact_id: str) -> Optional[Artifact]:
        """Asynchronous artifact loading."""
        try:
            async with aiofiles.open(artifact_id, "r") as f:
                content = await f.read()
                return Artifact.parse_raw(content)
        except Exception as e:
            print(f"Async artifact load error: {e}")
            return None
