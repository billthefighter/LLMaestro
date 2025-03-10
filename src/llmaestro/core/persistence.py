from pydantic import BaseModel


class PersistentModel(BaseModel):
    """Base class for Pydantic models that should be persisted to the database."""

    class Config:
        persistent = True
