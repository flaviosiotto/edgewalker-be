"""Rendered bodies for the transactional emails.

Each builder returns ``(subject, text_body, html_body)``. Every message is sent
as multipart with a plain-text alternative, because security mail that only
renders as HTML is the kind that lands in spam.
"""

from html import escape
from urllib.parse import quote, urljoin

from app.core.config import settings

_HTML_SHELL = """\
<!doctype html>
<html lang="it">
  <body style="margin:0;padding:24px;background:#f5f6f8;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#1f2933;">
    <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="max-width:520px;margin:0 auto;background:#ffffff;border-radius:12px;padding:32px;">
      <tr>
        <td>
          <h1 style="margin:0 0 16px;font-size:20px;font-weight:600;">{heading}</h1>
          {body}
          <p style="margin:32px 0 0;font-size:12px;line-height:18px;color:#7b8794;">
            Questo messaggio è stato inviato automaticamente da {product}. Non rispondere a questa email.
          </p>
        </td>
      </tr>
    </table>
  </body>
</html>
"""

_BUTTON = """\
<p style="margin:24px 0;">
  <a href="{url}" style="display:inline-block;background:#2563eb;color:#ffffff;text-decoration:none;padding:12px 20px;border-radius:8px;font-weight:600;">{label}</a>
</p>
<p style="margin:0 0 8px;font-size:13px;line-height:20px;color:#52606d;">
  Se il pulsante non funziona, copia questo indirizzo nel browser:
</p>
<p style="margin:0;font-size:13px;line-height:20px;word-break:break-all;">
  <a href="{url}" style="color:#2563eb;">{url}</a>
</p>
"""


def _paragraph(text: str) -> str:
    return f'<p style="margin:0 0 12px;font-size:15px;line-height:22px;">{escape(text)}</p>'


def _render(heading: str, body: str) -> str:
    return _HTML_SHELL.format(
        heading=escape(heading),
        body=body,
        product=escape(settings.EMAIL_FROM_NAME),
    )


def build_frontend_url(path: str, **query: str) -> str:
    base = settings.FRONTEND_BASE_URL.rstrip("/") + "/"
    url = urljoin(base, path.lstrip("/"))
    if query:
        encoded = "&".join(f"{key}={quote(value, safe='')}" for key, value in query.items())
        url = f"{url}?{encoded}"
    return url


def password_reset_email(*, display_name: str, reset_token: str, expires_minutes: int):
    reset_url = build_frontend_url("password-reset", token=reset_token)
    subject = f"Reimposta la tua password {settings.EMAIL_FROM_NAME}"

    text_body = (
        f"Ciao {display_name},\n\n"
        "abbiamo ricevuto una richiesta di reimpostazione della password del tuo account.\n\n"
        f"Apri questo link per scegliere una nuova password:\n{reset_url}\n\n"
        f"Il link vale {expires_minutes} minuti e può essere usato una sola volta.\n\n"
        "Se non hai richiesto tu il reset puoi ignorare questa email: la password attuale resta valida.\n"
    )

    html_body = _render(
        "Reimposta la tua password",
        _paragraph(f"Ciao {display_name},")
        + _paragraph(
            "abbiamo ricevuto una richiesta di reimpostazione della password del tuo account."
        )
        + _BUTTON.format(url=escape(reset_url, quote=True), label="Scegli una nuova password")
        + _paragraph(
            f"Il link vale {expires_minutes} minuti e può essere usato una sola volta."
        )
        + _paragraph(
            "Se non hai richiesto tu il reset puoi ignorare questa email: "
            "la password attuale resta valida."
        ),
    )

    return subject, text_body, html_body


def email_verification_email(*, display_name: str, verification_token: str, expires_hours: int):
    verify_url = build_frontend_url("verify-email", token=verification_token)
    subject = f"Conferma il tuo indirizzo email {settings.EMAIL_FROM_NAME}"

    text_body = (
        f"Ciao {display_name},\n\n"
        f"benvenuto in {settings.EMAIL_FROM_NAME}. Conferma il tuo indirizzo email "
        "per completare la registrazione.\n\n"
        f"Apri questo link:\n{verify_url}\n\n"
        f"Il link vale {expires_hours} ore.\n\n"
        "Se non ti sei registrato tu, ignora questa email.\n"
    )

    html_body = _render(
        "Conferma il tuo indirizzo email",
        _paragraph(f"Ciao {display_name},")
        + _paragraph(
            f"benvenuto in {settings.EMAIL_FROM_NAME}. Conferma il tuo indirizzo email "
            "per completare la registrazione."
        )
        + _BUTTON.format(url=escape(verify_url, quote=True), label="Conferma l'indirizzo")
        + _paragraph(f"Il link vale {expires_hours} ore.")
        + _paragraph("Se non ti sei registrato tu, ignora questa email."),
    )

    return subject, text_body, html_body


def account_pending_approval_email(*, display_name: str):
    subject = f"Richiesta di accesso a {settings.EMAIL_FROM_NAME} ricevuta"

    text_body = (
        f"Ciao {display_name},\n\n"
        "il tuo indirizzo email e' stato confermato.\n\n"
        f"{settings.EMAIL_FROM_NAME} e' in una fase ad accesso riservato, quindi il tuo "
        "account deve essere approvato da un amministratore. Riceverai una email "
        "non appena sara' attivo.\n"
    )

    html_body = _render(
        "Richiesta ricevuta",
        _paragraph(f"Ciao {display_name},")
        + _paragraph("il tuo indirizzo email e' stato confermato.")
        + _paragraph(
            f"{settings.EMAIL_FROM_NAME} e' in una fase ad accesso riservato, quindi il tuo "
            "account deve essere approvato da un amministratore. Riceverai una email "
            "non appena sara' attivo."
        ),
    )

    return subject, text_body, html_body


def account_approved_email(*, display_name: str):
    login_url = build_frontend_url("login")
    subject = f"Il tuo account {settings.EMAIL_FROM_NAME} e' attivo"

    text_body = (
        f"Ciao {display_name},\n\n"
        "il tuo account e' stato attivato. Puoi accedere da qui:\n"
        f"{login_url}\n"
    )

    html_body = _render(
        "Il tuo account e' attivo",
        _paragraph(f"Ciao {display_name},")
        + _paragraph("il tuo account e' stato attivato.")
        + _BUTTON.format(url=escape(login_url, quote=True), label="Accedi"),
    )

    return subject, text_body, html_body


def account_rejected_email(*, display_name: str):
    subject = f"Richiesta di accesso a {settings.EMAIL_FROM_NAME}"

    text_body = (
        f"Ciao {display_name},\n\n"
        "la tua richiesta di accesso non e' stata approvata.\n\n"
        "Se pensi si tratti di un errore, rispondi a chi ti ha invitato.\n"
    )

    html_body = _render(
        "Richiesta non approvata",
        _paragraph(f"Ciao {display_name},")
        + _paragraph("la tua richiesta di accesso non e' stata approvata.")
        + _paragraph("Se pensi si tratti di un errore, rispondi a chi ti ha invitato."),
    )

    return subject, text_body, html_body


def password_reset_for_external_account_email(*, display_name: str, provider_label: str):
    login_url = build_frontend_url("login")
    subject = f"Accesso al tuo account {settings.EMAIL_FROM_NAME}"

    text_body = (
        f"Ciao {display_name},\n\n"
        "abbiamo ricevuto una richiesta di reimpostazione password per il tuo account.\n\n"
        f"Il tuo account non ha una password: accedi con {provider_label}.\n"
        f"{login_url}\n\n"
        "Se non hai fatto tu la richiesta puoi ignorare questa email.\n"
    )

    html_body = _render(
        "Accedi con " + provider_label,
        _paragraph(f"Ciao {display_name},")
        + _paragraph(
            "abbiamo ricevuto una richiesta di reimpostazione password per il tuo account."
        )
        + _paragraph(f"Il tuo account non ha una password: accedi con {provider_label}.")
        + _BUTTON.format(url=escape(login_url, quote=True), label=f"Accedi con {provider_label}")
        + _paragraph("Se non hai fatto tu la richiesta puoi ignorare questa email."),
    )

    return subject, text_body, html_body


def password_changed_email(*, display_name: str):
    subject = f"La password del tuo account {settings.EMAIL_FROM_NAME} è stata modificata"

    text_body = (
        f"Ciao {display_name},\n\n"
        "la password del tuo account è stata modificata.\n\n"
        "Se non sei stato tu, contatta subito un amministratore: qualcuno potrebbe "
        "avere accesso al tuo account.\n"
    )

    html_body = _render(
        "Password modificata",
        _paragraph(f"Ciao {display_name},")
        + _paragraph("la password del tuo account è stata modificata.")
        + _paragraph(
            "Se non sei stato tu, contatta subito un amministratore: "
            "qualcuno potrebbe avere accesso al tuo account."
        ),
    )

    return subject, text_body, html_body
