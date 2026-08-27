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


class SkillPrivacyError(SkillError):
    """The active privacy mode hides persistent procedural skills."""


class EnablementUnavailableError(SkillError):
    """The agent database required for enablement is unavailable."""


class GitSourceError(SkillError):
    """A git-backed source could not be validated or read."""


class GitInputError(GitSourceError, ValueError):
    """A caller supplied an invalid Git URL or ref."""


__all__ = [
    "EnablementUnavailableError",
    "GitInputError",
    "GitSourceError",
    "SkillConflictError",
    "SkillError",
    "SkillFormatError",
    "SkillNotFoundError",
    "SkillPathError",
    "SkillPrivacyError",
    "SkillReadOnlyError",
]
