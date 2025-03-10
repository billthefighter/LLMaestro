"""Core attachment handling for file and image content."""

from abc import ABC, abstractmethod
from pathlib import Path
import base64
from typing import Union, Optional, Dict, Any

from pydantic import Field, ConfigDict
from llmaestro.llm.enums import MediaType
from llmaestro.core.persistence import PersistentModel


class BaseAttachment(PersistentModel, ABC):
    """Base class for all file attachments."""

    content: Union[str, bytes] = Field(description="The file content, either as string or bytes")
    media_type: MediaType = Field(description="The media type of the file")
    file_name: Optional[str] = Field(default=None, description="Name of the file")
    description: Optional[str] = Field(default=None, description="Optional description of the file")

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "BaseAttachment":
        """Create an attachment from a file path."""
        path = Path(file_path)
        with open(path, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        media_type = MediaType.from_file_extension(str(path))
        return cls(content=content, media_type=media_type, file_name=path.name)

    @classmethod
    def from_bytes(
        cls, data: bytes, mime_type: Union[str, MediaType], file_name: Optional[str] = None
    ) -> "BaseAttachment":
        """Create an attachment from bytes data."""
        content = base64.b64encode(data).decode()
        media_type = mime_type if isinstance(mime_type, MediaType) else MediaType.from_mime_type(mime_type)
        return cls(content=content, media_type=media_type, file_name=file_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert attachment to a dictionary format."""
        return {
            "content": self.content,
            "mime_type": str(self.media_type),
            "file_name": self.file_name or "",
            "description": self.description,
        }

    @abstractmethod
    def validate(self) -> bool:
        """Validate the attachment content and media type.

        Returns:
            bool: True if the attachment is valid, False otherwise.

        Raises:
            ValueError: If the attachment is invalid with a specific reason.
        """
        pass

    @abstractmethod
    def get_size(self) -> int:
        """Get the size of the attachment content in bytes.

        Returns:
            int: Size of the attachment in bytes.
        """
        pass

    @property
    @abstractmethod
    def content_type(self) -> str:
        """Get a string describing the type of content this attachment represents.

        Returns:
            str: Description of the content type (e.g., "image", "document", etc.)
        """
        pass


class ImageAttachment(BaseAttachment):
    """Specialized attachment for images with image-specific validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.validate()

    def validate(self) -> bool:
        """Validate that this is a valid image attachment."""
        if not self.media_type.is_image():
            raise ValueError(f"Invalid media type for image: {self.media_type}")
        return True

    def get_size(self) -> int:
        """Get the size of the image in bytes."""
        if isinstance(self.content, str):
            # If base64 encoded
            padding = len(self.content) % 4
            return (len(self.content) * 3) // 4 - padding
        return len(self.content)

    @property
    def content_type(self) -> str:
        """Get the content type."""
        return "image"


class FileAttachment(BaseAttachment):
    """General purpose file attachment."""

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)
    file_id: Optional[str] = Field(default=None, description="ID of the file when uploaded to an LLM provider")

    def validate(self) -> bool:
        """Validate the file attachment."""
        if not self.content:
            raise ValueError("File content cannot be empty")
        if not self.media_type:
            raise ValueError("Media type must be specified")
        return True

    def get_size(self) -> int:
        """Get the size of the file in bytes."""
        if isinstance(self.content, str):
            # If base64 encoded
            padding = len(self.content) % 4
            return (len(self.content) * 3) // 4 - padding
        return len(self.content)

    @property
    def content_type(self) -> str:
        """Get the content type."""
        if self.media_type.is_image():
            return "image"
        if self.media_type.is_document():
            return "document"
        return "file"


class AttachmentConverter:
    """Utility class for converting attachments between formats."""

    @staticmethod
    def to_interface_format(attachment: BaseAttachment) -> dict:
        """Convert attachment to format expected by LLM interfaces."""
        result = {
            "content": attachment.content,
            "media_type": str(attachment.media_type),
            "file_name": attachment.file_name or "",
        }
        # Add file_id if present
        if isinstance(attachment, FileAttachment) and attachment.file_id is not None:
            result["file_id"] = attachment.file_id
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> BaseAttachment:
        """Create an attachment from a dictionary format."""
        media_type = data.get("media_type") or data.get("mime_type")
        if not media_type:
            raise ValueError("media_type is required")

        media_type_enum = MediaType.from_mime_type(media_type)
        if media_type_enum.is_image():
            return ImageAttachment(content=data["content"], media_type=media_type_enum, file_name=data.get("file_name"))

        # Handle file attachments with potential file_id
        return FileAttachment(
            content=data["content"],
            media_type=media_type_enum,
            file_name=data.get("file_name"),
            file_id=data.get("file_id"),
        )
