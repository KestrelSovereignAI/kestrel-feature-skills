"""Error vocabulary for procedural skill operations."""


class SkillError(Exception):
    """Base class for visible, expected skill failures."""


class SkillFormatError(SkillError, ValueError):
    """A skill document or folder violates the public format."""


class SkillPathError(SkillError, ValueError):
    """A requested path escapes or violates a skill root."""


class SkillConflictError(SkillError):
    """A create or install conflicts with an existing skill."""


class SkillNotFoundError(SkillError):
    """A requested skill or file does not exist."""


class SkillReadOnlyError(SkillError):
    """A mutation targeted a non-local source."""


class EnablementUnavailableError(SkillError):
    """The agent database required for enablement is unavailable."""


class GitSourceError(SkillError):
    """A git-backed source could not be validated or read."""


__all__ = [
    "EnablementUnavailableError",
    "GitSourceError",
    "SkillConflictError",
    "SkillError",
    "SkillFormatError",
    "SkillNotFoundError",
    "SkillPathError",
    "SkillReadOnlyError",
]
