"""Exception hierarchy for reframe."""


class ReframeError(Exception):
    """Base exception for all reframe errors."""

    exit_code: int = 1

    def __init__(self, message: str = "", code: str = "reframe_error"):
        self.code = code
        self.message = message if message else code
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"ERROR [{self.code}] {self.message}"


class InvalidInputError(ReframeError):
    """Raised when input file, format, duration or arguments are invalid."""

    exit_code: int = 2

    def __init__(self, message: str = "", code: str = "invalid_input"):
        super().__init__(message=message or "invalid input", code=code)


class MissingDependencyError(ReframeError):
    """Raised when an external tool (ffmpeg/ffprobe) or model is missing."""

    exit_code: int = 3

    def __init__(self, message: str = "", code: str = "missing_dependency"):
        super().__init__(message=message or "missing dependency", code=code)


class ProcessingError(ReframeError):
    """Raised when processing fails during analysis or rendering."""

    exit_code: int = 4

    def __init__(self, message: str = "", code: str = "processing_error"):
        if not message:
            message = "not implemented" if code == "not_implemented" else (code or "processing error")
        super().__init__(message=message, code=code)


class OutputValidationError(ReframeError):
    """Raised when output validation fails."""

    exit_code: int = 5

    def __init__(self, message: str = "", code: str = "output_validation_failed"):
        super().__init__(message=message or "output validation failed", code=code)


__all__ = [
    "ReframeError",
    "InvalidInputError",
    "MissingDependencyError",
    "ProcessingError",
    "OutputValidationError",
]
