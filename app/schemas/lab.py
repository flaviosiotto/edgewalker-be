from pydantic import BaseModel


class LabLaunch(BaseModel):
    """One-shot launch URL for the embedded Studio Lab (JupyterHub)."""

    lab_url: str
