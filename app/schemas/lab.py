from typing import Optional

from pydantic import BaseModel, field_validator


class LabLaunchRequest(BaseModel):
    """Options for launching the embedded Studio Lab session."""

    # Path (relative to the user's workspace) to land on after login,
    # e.g. "notebooks/studios/stagionalita-btc-3.ipynb". None -> file browser.
    next_path: Optional[str] = None
    # Stop the running server first: the respawn re-syncs the studios into
    # the workspace (needed when a studio was created after the session
    # started). No-op when the hub API token is not configured.
    fresh: bool = False
    # App theme at launch time ("dark" | "light"): the spawn hook sets the
    # matching Jupyter theme so the embedded notebook blends in.
    theme: Optional[str] = None

    @field_validator("theme")
    @classmethod
    def _known_theme(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in ("dark", "light"):
            raise ValueError("theme must be 'dark' or 'light'")
        return value

    @field_validator("next_path")
    @classmethod
    def _safe_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if value.startswith("/") or ".." in value or "?" in value or "#" in value:
            raise ValueError("next_path must be a plain relative path")
        return value


class LabLaunch(BaseModel):
    """One-shot launch URL for the embedded Studio Lab (JupyterHub)."""

    lab_url: str


class LabThemeRequest(BaseModel):
    """Switch the running Jupyter session's theme to match the app."""

    theme: str

    @field_validator("theme")
    @classmethod
    def _known_theme(cls, value: str) -> str:
        if value not in ("dark", "light"):
            raise ValueError("theme must be 'dark' or 'light'")
        return value


class LabWorkspaceFile(BaseModel):
    """Raw text content of a file in the user's Lab workspace."""

    path: str
    content: str
    # ISO timestamp dal contents API: il FE lo confronta con la versione
    # corrente per non "fotografare" file mai toccati (stale del sync).
    last_modified: Optional[str] = None


class StudioAccessToken(BaseModel):
    """Short-lived credential for studio-svc, exchanged from a PAT/UI session."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    scopes: list[str]
