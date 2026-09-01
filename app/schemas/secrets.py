from datetime import datetime

from pydantic import BaseModel, Field


class SecretWrite(BaseModel):
    value: str = Field(min_length=1)


class SecretRead(BaseModel):
    name: str
    created_at: datetime
    updated_at: datetime
