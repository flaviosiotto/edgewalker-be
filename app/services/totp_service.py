"""Time-based one-time passwords (TOTP) and single-use recovery codes.

Secrets are encrypted at rest with Fernet. The key comes from
``MFA_ENCRYPTION_KEY`` when set, otherwise it is derived from ``SECRET_KEY``:
rotating whichever one is in use makes every enrolled authenticator
undecryptable, and users have to enrol again.

An enrolment row with ``confirmed_at IS NULL`` is a started-but-unproven setup.
It must never gate login, or a failed enrolment would lock the account out.
"""

import base64
import hashlib
import io
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import pyotp
import qrcode
import qrcode.image.svg
from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.core.config import settings
from app.models.access_control import UserRecoveryCode, UserTotp
from app.models.user import User
from app.utils.auth_utils import create_delegated_token, decode_token, verify_password

logger = logging.getLogger(__name__)

MFA_CHALLENGE_PURPOSE = "mfa_challenge"

# Accept the adjacent 30s steps so a slightly skewed phone clock still works.
_TOTP_VALID_WINDOW = 1


def _fernet() -> Fernet:
    material = settings.MFA_ENCRYPTION_KEY or settings.SECRET_KEY
    if not material or material == "change-me":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La configurazione della verifica in due passaggi non e' completa.",
        )
    key = base64.urlsafe_b64encode(hashlib.sha256(material.encode("utf-8")).digest())
    return Fernet(key)


def _encrypt_secret(secret: str) -> str:
    return _fernet().encrypt(secret.encode("utf-8")).decode("ascii")


def _decrypt_secret(stored: str) -> str:
    try:
        return _fernet().decrypt(stored.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        logger.error("Stored TOTP secret could not be decrypted; was the key rotated?")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Impossibile leggere la configurazione 2FA di questo account.",
        ) from exc


def _hash_recovery_code(code: str) -> str:
    return hashlib.sha256(code.replace("-", "").upper().encode("utf-8")).hexdigest()


def get_enrolment(session: Session, user_id: int) -> Optional[UserTotp]:
    return session.get(UserTotp, user_id)


def is_mfa_enabled(session: Session, user_id: int) -> bool:
    enrolment = get_enrolment(session, user_id)
    return enrolment is not None and enrolment.confirmed_at is not None


def start_enrollment(session: Session, user: User) -> dict[str, str]:
    """Create (or replace) an unconfirmed enrolment and return its QR code."""
    if is_mfa_enabled(session, user.id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="La verifica in due passaggi e' gia' attiva.",
        )

    existing = get_enrolment(session, user.id)
    if existing is not None:
        # Restarting setup issues a brand new secret, so an abandoned attempt
        # cannot be confirmed later by whoever scanned that older QR code.
        session.delete(existing)
        session.commit()

    secret = pyotp.random_base32()
    session.add(
        UserTotp(
            user_id=user.id,
            secret=_encrypt_secret(secret),
        )
    )
    session.commit()

    provisioning_uri = pyotp.TOTP(secret).provisioning_uri(
        name=user.email,
        issuer_name=settings.MFA_ISSUER_NAME,
    )

    return {
        "secret": secret,
        "otpauth_uri": provisioning_uri,
        "qr_svg": _render_qr_svg(provisioning_uri),
    }


def _render_qr_svg(data: str) -> str:
    image = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage)
    buffer = io.BytesIO()
    image.save(buffer)
    return buffer.getvalue().decode("utf-8")


def _generate_recovery_codes(count: int) -> list[str]:
    codes = []
    for _ in range(count):
        raw = base64.b32encode(secrets.token_bytes(10)).decode("ascii").rstrip("=")[:10]
        codes.append(f"{raw[:5]}-{raw[5:]}")
    return codes


def confirm_enrollment(session: Session, user: User, code: str) -> list[str]:
    """Prove the authenticator works, activate 2FA and issue recovery codes."""
    enrolment = get_enrolment(session, user.id)
    if enrolment is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Nessuna configurazione 2FA da confermare. Ricomincia la procedura.",
        )
    if enrolment.confirmed_at is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="La verifica in due passaggi e' gia' attiva.",
        )

    secret = _decrypt_secret(enrolment.secret)
    if not pyotp.TOTP(secret).verify(code.strip(), valid_window=_TOTP_VALID_WINDOW):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Codice non valido. Controlla l'app di autenticazione e riprova.",
        )

    enrolment.confirmed_at = datetime.now(timezone.utc)
    session.add(enrolment)

    codes = _generate_recovery_codes(settings.MFA_RECOVERY_CODE_COUNT)
    for code_value in codes:
        session.add(
            UserRecoveryCode(user_id=user.id, code_hash=_hash_recovery_code(code_value))
        )
    session.commit()

    # Returned exactly once: only their hashes are stored.
    return codes


def disable_mfa(session: Session, user: User, password: Optional[str], code: Optional[str]) -> None:
    """Turn 2FA off, after re-proving identity.

    Requires the current password, or a valid TOTP/recovery code for accounts
    that sign in through an external provider and have no password.
    """
    if not is_mfa_enabled(session, user.id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="La verifica in due passaggi non e' attiva.",
        )

    reauthenticated = False
    if password and user.has_usable_password:
        reauthenticated = verify_password(password, user.hashed_password)
    if not reauthenticated and code:
        reauthenticated = verify_mfa_code(session, user, code)

    if not reauthenticated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conferma con la password attuale o con un codice valido.",
        )

    enrolment = get_enrolment(session, user.id)
    if enrolment is not None:
        session.delete(enrolment)
    for recovery in session.exec(
        select(UserRecoveryCode).where(UserRecoveryCode.user_id == user.id)
    ).all():
        session.delete(recovery)
    session.commit()


def verify_mfa_code(session: Session, user: User, code: str) -> bool:
    """Accept either a current TOTP code or an unused recovery code."""
    candidate = code.strip()
    if not candidate:
        return False

    enrolment = get_enrolment(session, user.id)
    if enrolment is None or enrolment.confirmed_at is None:
        return False

    secret = _decrypt_secret(enrolment.secret)
    if pyotp.TOTP(secret).verify(candidate, valid_window=_TOTP_VALID_WINDOW):
        return True

    recovery = session.exec(
        select(UserRecoveryCode).where(
            UserRecoveryCode.user_id == user.id,
            UserRecoveryCode.code_hash == _hash_recovery_code(candidate),
            UserRecoveryCode.used_at.is_(None),
        )
    ).first()
    if recovery is None:
        return False

    # Burn it: a recovery code works once.
    recovery.used_at = datetime.now(timezone.utc)
    session.add(recovery)
    session.commit()
    return True


def count_unused_recovery_codes(session: Session, user_id: int) -> int:
    return len(
        session.exec(
            select(UserRecoveryCode).where(
                UserRecoveryCode.user_id == user_id,
                UserRecoveryCode.used_at.is_(None),
            )
        ).all()
    )


def issue_mfa_challenge_token(user: User) -> str:
    """Short-lived token proving the first factor was already satisfied."""
    return create_delegated_token(
        {"sub": user.email, "uid": user.id},
        audience=settings.MFA_TOKEN_AUDIENCE,
        purpose=MFA_CHALLENGE_PURPOSE,
        expires_delta=timedelta(minutes=settings.MFA_CHALLENGE_EXPIRE_MINUTES),
    )


def resolve_mfa_challenge_token(session: Session, token: str) -> User:
    payload = decode_token(token, audience=settings.MFA_TOKEN_AUDIENCE)
    if payload is None or payload.get("purpose") != MFA_CHALLENGE_PURPOSE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sessione di verifica scaduta. Rifai il login.",
        )

    user_id = payload.get("uid")
    user = session.get(User, user_id) if user_id is not None else None
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sessione di verifica non valida.",
        )
    return user
