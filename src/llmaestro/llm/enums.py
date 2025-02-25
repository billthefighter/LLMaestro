"""Common enums for LLM functionality."""
import mimetypes
from enum import Enum


class ModelFamily(str, Enum):
    """Canonical provider definitions."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

    @property
    def display_name(self) -> str:
        """Get a human-readable display name."""
        return {
            self.ANTHROPIC: "Anthropic",
            self.OPENAI: "OpenAI",
            self.GOOGLE: "Google",
            self.HUGGINGFACE: "Hugging Face",
            self.CUSTOM: "Custom",
        }[self]

    @property
    def default_api_base(self) -> str:
        """Get the default API base URL for this provider."""
        return {
            self.ANTHROPIC: "https://api.anthropic.com/v1",
            self.OPENAI: "https://api.openai.com/v1",
            self.GOOGLE: "https://generativelanguage.googleapis.com/v1",
            self.HUGGINGFACE: "https://api.huggingface.co",
            self.CUSTOM: "",
        }[self]

    @classmethod
    def from_name(cls, name: str) -> "ModelFamily":
        """Get a provider from a name (case-insensitive)."""
        try:
            return cls(name.lower())
        except ValueError:
            return cls.CUSTOM


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
