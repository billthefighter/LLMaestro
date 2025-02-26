"""Common enums for LLM functionality."""
import mimetypes
from enum import Enum


class MediaType(str, Enum):
    """Standard media types for LLM inputs."""

    # Image formats
    JPEG = "image/jpeg"
    PNG = "image/png"
    GIF = "image/gif"
    WEBP = "image/webp"
    BMP = "image/bmp"
    TIFF = "image/tiff"
    SVG = "image/svg+xml"

    # Document formats
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    DOC = "application/msword"
    TXT = "text/plain"

    # Fallback
    UNKNOWN = "application/octet-stream"

    @classmethod
    def from_mime_type(cls, mime_type: str) -> "MediaType":
        """Convert a MIME type string to MediaType enum."""
        try:
            return cls(mime_type)
        except ValueError:
            return cls.UNKNOWN

    @classmethod
    def from_file_extension(cls, file_path: str) -> "MediaType":
        """Detect media type from file extension."""
        mime_type = mimetypes.guess_type(str(file_path))[0]
        return cls.from_mime_type(mime_type or "application/octet-stream")

    def is_image(self) -> bool:
        """Check if this media type represents an image format."""
        return self.value.startswith("image/")

    def is_document(self) -> bool:
        """Check if this media type represents a document format."""
        return self.value.startswith("application/")

    def __str__(self) -> str:
        return self.value
