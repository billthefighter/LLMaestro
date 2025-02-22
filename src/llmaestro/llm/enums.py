"""Common enums for LLM functionality."""
import mimetypes
from enum import Enum


class ModelFamily(str, Enum):
    """Supported model families with their provider associations."""

    CLAUDE = "claude"
    GPT = "gpt"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

    @classmethod
    def from_provider(cls, provider_name: str) -> "ModelFamily":
        """Get the model family for a provider name.

        Args:
            provider_name: Name of the provider (case-insensitive)

        Returns:
            Corresponding ModelFamily or CUSTOM if provider not recognized
        """
        mapping = {
            "anthropic": cls.CLAUDE,
            "openai": cls.GPT,
            "google": cls.GEMINI,
            "huggingface": cls.HUGGINGFACE,
        }
        return mapping.get(provider_name.lower(), cls.CUSTOM)

    @property
    def provider_name(self) -> str:
        """Get the canonical provider name for this family.

        Returns:
            Provider name or 'custom' if no specific provider
        """
        mapping = {
            self.CLAUDE: "anthropic",
            self.GPT: "openai",
            self.GEMINI: "google",
            self.HUGGINGFACE: "huggingface",
        }
        return mapping.get(self, "custom")

    @property
    def display_name(self) -> str:
        """Get a human-readable display name for this family.

        Returns:
            Display name for the family
        """
        mapping = {
            self.CLAUDE: "Anthropic",
            self.GPT: "OpenAI",
            self.GEMINI: "Google",
            self.HUGGINGFACE: "Hugging Face",
            self.CUSTOM: "Custom",
        }
        return mapping[self]


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
