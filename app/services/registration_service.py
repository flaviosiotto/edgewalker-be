"""Self-service registration and the family&friends access gate.

The gate lives in :func:`resolve_signup_outcome` and is deliberately the single
place that decides whether a brand new account becomes usable. Both the local
signup and the external (OAuth) callback must call it: if the gate lived only in
the local path, "sign in with Google" would be a way around the invite phase.
"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import BackgroundTasks, HTTPException, status
from sqlmodel import Session, select

from app.core.config import settings
from app.models.access_control import AccessAllowlist, EmailVerificationToken
from app.models.user import User, UserStatus, apply_status
from app.schemas.auth import RegistrationRequest
from app.services.email_service import queue_email
from app.services.email_templates import (
    account_approved_email,
    account_pending_approval_email,
    account_rejected_email,
    email_verification_email,
)
from app.services.password_reset_service import validate_password_strength
from app.utils.auth_utils import get_password_hash

logger = logging.getLogger(__name__)

# Deliberately identical whether or not the address is already registered, so
# the endpoint cannot be used to enumerate accounts.
GENERIC_REGISTRATION_MESSAGE = (
    "Se l'indirizzo e' utilizzabile, ti abbiamo inviato una email per confermarlo."
)


class RegistrationMode:
    CLOSED = "closed"
    FAMILY_AND_FRIENDS = "family_and_friends"
    OPEN = "open"


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def find_allowlist_entry(session: Session, email: str) -> Optional[AccessAllowlist]:
    statement = select(AccessAllowlist).where(AccessAllowlist.email == normalize_email(email))
    return session.exec(statement).first()


def resolve_signup_outcome(session: Session, email: str) -> UserStatus:
    """Decide the status a newly verified account lands in.

    Returns the status to apply once the address is confirmed. Raises 403 when
    the configured mode forbids the signup outright.

    Called by BOTH the local registration flow and the external identity flow.
    """
    mode = settings.REGISTRATION_MODE
    allowlisted = find_allowlist_entry(session, email) is not None

    if mode == RegistrationMode.OPEN:
        return UserStatus.ACTIVE

    if allowlisted:
        return UserStatus.ACTIVE

    if mode == RegistrationMode.CLOSED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Le registrazioni sono attualmente chiuse.",
        )

    if mode != RegistrationMode.FAMILY_AND_FRIENDS:
        # An unrecognised value must not silently widen access.
        logger.error(
            "Unknown REGISTRATION_MODE %r, refusing signup for safety", mode
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Le registrazioni sono attualmente chiuse.",
        )

    return UserStatus.PENDING_APPROVAL


def consume_allowlist_entry(session: Session, user: User) -> None:
    entry = find_allowlist_entry(session, user.email)
    if entry is None or entry.consumed_at is not None:
        return
    entry.consumed_at = datetime.now(timezone.utc)
    entry.consumed_by_user_id = user.id
    session.add(entry)


def _invalidate_verification_tokens(session: Session, user_id: int, used_at: datetime) -> None:
    statement = select(EmailVerificationToken).where(
        EmailVerificationToken.user_id == user_id,
        EmailVerificationToken.used_at.is_(None),
    )
    for token in session.exec(statement).all():
        token.used_at = used_at
        session.add(token)


def issue_verification_email(
    session: Session,
    user: User,
    background_tasks: Optional[BackgroundTasks] = None,
) -> None:
    """Replace any outstanding verification token and email the new one."""
    now = datetime.now(timezone.utc)
    _invalidate_verification_tokens(session, user.id, now)

    raw_token = secrets.token_urlsafe(32)
    expires_at = now + timedelta(hours=settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS)
    session.add(
        EmailVerificationToken(
            user_id=user.id,
            token_hash=_hash_token(raw_token),
            expires_at=expires_at,
        )
    )

    recipient_email, display_name = user.email, user.display_name
    session.commit()

    subject, text_body, html_body = email_verification_email(
        display_name=display_name,
        verification_token=raw_token,
        expires_hours=settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS,
    )
    queue_email(
        background_tasks,
        to_address=recipient_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )


def register_user(
    session: Session,
    payload: RegistrationRequest,
    background_tasks: Optional[BackgroundTasks] = None,
) -> dict[str, object]:
    """Create a pending account and send the confirmation email.

    The response never reveals whether the address was already taken.
    """
    email = normalize_email(payload.email)
    username = payload.username.strip()

    if not username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username obbligatorio"
        )

    validate_password_strength(payload.password)

    # Reject up front in closed mode so a non-invited address never creates a
    # row, but only after validating the payload so the error is about the
    # right thing.
    if settings.REGISTRATION_MODE == RegistrationMode.CLOSED:
        if find_allowlist_entry(session, email) is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Le registrazioni sono attualmente chiuse.",
            )

    existing_email = session.exec(select(User).where(User.email == email)).first()
    if existing_email is not None:
        # Same wording as the success path. Re-send the verification link if the
        # account never got confirmed, so an abandoned signup can be resumed.
        if existing_email.status == UserStatus.PENDING_EMAIL.value:
            issue_verification_email(session, existing_email, background_tasks)
        return {"message": GENERIC_REGISTRATION_MESSAGE, "status": UserStatus.PENDING_EMAIL.value}

    existing_username = session.exec(select(User).where(User.username == username)).first()
    if existing_username is not None:
        # The username is public-ish and has a unique constraint, so a distinct
        # error here is not an account-enumeration leak on the email.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Username già in uso"
        )

    user = User(
        email=email,
        username=username,
        first_name=(payload.first_name or "").strip() or None,
        last_name=(payload.last_name or "").strip() or None,
        hashed_password=get_password_hash(payload.password),
        role="user",
    )
    apply_status(user, UserStatus.PENDING_EMAIL)
    session.add(user)
    session.commit()
    session.refresh(user)

    issue_verification_email(session, user, background_tasks)

    return {"message": GENERIC_REGISTRATION_MESSAGE, "status": UserStatus.PENDING_EMAIL.value}


def resend_verification_email(
    session: Session,
    email: str,
    background_tasks: Optional[BackgroundTasks] = None,
) -> dict[str, object]:
    user = session.exec(select(User).where(User.email == normalize_email(email))).first()
    if user is not None and user.status == UserStatus.PENDING_EMAIL.value:
        issue_verification_email(session, user, background_tasks)
    return {"message": GENERIC_REGISTRATION_MESSAGE, "status": UserStatus.PENDING_EMAIL.value}


def confirm_email_verification(
    session: Session,
    token: str,
    background_tasks: Optional[BackgroundTasks] = None,
) -> dict[str, object]:
    """Confirm the address, then apply the access gate."""
    normalized_token = token.strip()
    if not normalized_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Token di verifica obbligatorio"
        )

    now = datetime.now(timezone.utc)
    statement = select(EmailVerificationToken).where(
        EmailVerificationToken.token_hash == _hash_token(normalized_token),
        EmailVerificationToken.used_at.is_(None),
        EmailVerificationToken.expires_at > now,
    )
    verification = session.exec(statement).first()
    if verification is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Il link di verifica non e' valido o e' scaduto",
        )

    user = session.get(User, verification.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Utente non disponibile"
        )

    verification.used_at = now
    session.add(verification)

    # Verifying again after approval must not knock the account back to pending.
    if user.status != UserStatus.PENDING_EMAIL.value:
        if user.email_verified_at is None:
            user.email_verified_at = now
            session.add(user)
        session.commit()
        return {"status": user.status, "message": _status_message(user.status)}

    user.email_verified_at = now
    outcome = resolve_signup_outcome(session, user.email)
    apply_status(user, outcome)

    if outcome == UserStatus.ACTIVE:
        user.approved_at = now
        consume_allowlist_entry(session, user)

    recipient_email, display_name = user.email, user.display_name
    session.add(user)
    session.commit()

    if outcome == UserStatus.ACTIVE:
        subject, text_body, html_body = account_approved_email(display_name=display_name)
    else:
        subject, text_body, html_body = account_pending_approval_email(display_name=display_name)
    queue_email(
        background_tasks,
        to_address=recipient_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )

    return {"status": outcome.value, "message": _status_message(outcome.value)}


def _status_message(status_value: str) -> str:
    if status_value == UserStatus.ACTIVE.value:
        return "Email confermata. Il tuo account e' attivo: puoi accedere."
    if status_value == UserStatus.PENDING_APPROVAL.value:
        return (
            "Email confermata. Il tuo account e' in attesa di approvazione da parte "
            "di un amministratore: riceverai una email quando sara' attivo."
        )
    if status_value == UserStatus.REJECTED.value:
        return "La richiesta di accesso e' stata rifiutata."
    if status_value == UserStatus.SUSPENDED.value:
        return "Questo account e' sospeso."
    return "Email confermata."


# ---------------------------------------------------------------------------
# Administrator actions
# ---------------------------------------------------------------------------


def approve_user(
    session: Session,
    user_id: int,
    admin: User,
    background_tasks: Optional[BackgroundTasks] = None,
) -> User:
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Utente non trovato")

    if user.status == UserStatus.ACTIVE.value:
        return user

    if user.status == UserStatus.PENDING_EMAIL.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="L'utente non ha ancora confermato l'indirizzo email",
        )

    now = datetime.now(timezone.utc)
    apply_status(user, UserStatus.ACTIVE)
    user.approved_at = now
    user.approved_by = admin.id
    consume_allowlist_entry(session, user)

    recipient_email, display_name = user.email, user.display_name
    session.add(user)
    session.commit()
    session.refresh(user)

    subject, text_body, html_body = account_approved_email(display_name=display_name)
    queue_email(
        background_tasks,
        to_address=recipient_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )
    return user


def reject_user(
    session: Session,
    user_id: int,
    admin: User,
    background_tasks: Optional[BackgroundTasks] = None,
    notify: bool = True,
) -> User:
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Utente non trovato")

    if user.role == "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Non e' possibile rifiutare un amministratore",
        )

    apply_status(user, UserStatus.REJECTED)
    user.approved_by = admin.id

    recipient_email, display_name = user.email, user.display_name
    session.add(user)
    session.commit()
    session.refresh(user)

    if notify:
        subject, text_body, html_body = account_rejected_email(display_name=display_name)
        queue_email(
            background_tasks,
            to_address=recipient_email,
            subject=subject,
            text_body=text_body,
            html_body=html_body,
        )
    return user


def list_users_with_status(session: Session, status_filter: Optional[str] = None) -> list[User]:
    statement = select(User)
    if status_filter:
        statement = statement.where(User.status == status_filter)
    return list(session.exec(statement.order_by(User.created_at.desc())).all())


def add_allowlist_entry(
    session: Session, email: str, note: Optional[str], admin: User
) -> AccessAllowlist:
    normalized = normalize_email(email)
    existing = find_allowlist_entry(session, normalized)
    if existing is not None:
        return existing

    entry = AccessAllowlist(email=normalized, note=note, invited_by=admin.id)
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry


def list_allowlist(session: Session) -> list[AccessAllowlist]:
    statement = select(AccessAllowlist).order_by(AccessAllowlist.created_at.desc())
    return list(session.exec(statement).all())


def delete_allowlist_entry(session: Session, entry_id: int) -> None:
    entry = session.get(AccessAllowlist, entry_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Voce di allowlist non trovata"
        )
    session.delete(entry)
    session.commit()
