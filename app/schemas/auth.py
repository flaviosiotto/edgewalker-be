from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenWithRefresh(Token):
    refresh_token: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class PasswordResetRequest(BaseModel):
    identifier: str


class PasswordResetConfirmRequest(BaseModel):
    token: str
    new_password: str


class PasswordResetRequestResponse(BaseModel):
    message: str
    reset_token: str | None = None
    expires_at: datetime | None = None


class MessageResponse(BaseModel):
    message: str


class AuthProvidersResponse(BaseModel):
    google: bool


class LoginResponse(BaseModel):
    """Either a token pair, or a challenge when the account has 2FA enabled.

    Accounts without 2FA keep seeing exactly the previous shape, so existing
    clients are unaffected.
    """

    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: Optional[str] = None
    mfa_required: bool = False
    mfa_token: Optional[str] = None


class MfaVerifyRequest(BaseModel):
    mfa_token: str
    code: str


class MfaStatusResponse(BaseModel):
    enabled: bool
    recovery_codes_remaining: int = 0


class MfaSetupResponse(BaseModel):
    secret: str
    otpauth_uri: str
    qr_svg: str


class MfaConfirmRequest(BaseModel):
    code: str


class MfaRecoveryCodesResponse(BaseModel):
    recovery_codes: list[str]


class MfaDisableRequest(BaseModel):
    password: Optional[str] = None
    code: Optional[str] = None


class OAuthExchangeRequest(BaseModel):
    code: str


class RegistrationRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=64)
    password: str
    first_name: Optional[str] = Field(default=None, max_length=120)
    last_name: Optional[str] = Field(default=None, max_length=120)


class RegistrationResponse(BaseModel):
    message: str
    status: str


class EmailVerificationRequest(BaseModel):
    token: str


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class EmailVerificationResponse(BaseModel):
    status: str
    message: str


class AllowlistCreateRequest(BaseModel):
    email: EmailStr
    note: Optional[str] = None


class AllowlistEntryRead(BaseModel):
    id: int
    email: str
    note: Optional[str] = None
    invited_by: Optional[int] = None
    created_at: datetime
    consumed_at: Optional[datetime] = None
    consumed_by_user_id: Optional[int] = None


class PendingUserRead(BaseModel):
    id: int
    email: str
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: str
    status: str
    is_active: bool
    created_at: datetime
    email_verified_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
