import logging

from fastapi import APIRouter, Depends, Query, Response, status
from sqlmodel import Session

from app.db.database import get_session
from app.models.user import User
from app.schemas.scan import ScanCreate, ScanListPage, ScanRead, ScanRowRead, ScanRowsPage, ScanSummary
from app.services import scan_service
from app.utils.auth_utils import get_current_active_or_consultative_user

logger = logging.getLogger(__name__)

# Symbol scans (docs/valutazione-scanner-simboli.md): a strategy's indicators
# and rule conditions read on the last closed bar of many symbols. Auth mirrors
# /backtests (user JWT, PAT, the agent's consultative token): a hosted agent
# launches a scan from the strategy chat the way it launches a backtest.
# No endpoint ranks the rows or names a "best" symbol: the order is by symbol
# unless the caller asks for a sort key.
router = APIRouter(prefix="/scans", tags=["Scans"])


def _read(scan) -> ScanRead:
    read = ScanRead.model_validate(scan, from_attributes=True)
    read.free = isinstance(scan.request, dict) and isinstance(scan.request.get("free"), dict)
    read.params = scan_service.scan_params(scan.config)
    read.source, read.template_id = scan_service.scan_source(scan.request)
    return read


@router.post("/", response_model=ScanRead, status_code=status.HTTP_201_CREATED)
def create_scan_endpoint(
    payload: ScanCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Create a scan and start it: from a strategy (``strategy_id``), or free
    (``indicators`` and optional ``conditions`` on a ``connection_id``). The
    rows fill in while it runs: poll ``GET /scans/{id}/rows``."""
    scan = scan_service.create_scan(session, payload, current_user.id)
    return _read(scan)


@router.get("/", response_model=ScanListPage)
def list_scans_endpoint(
    strategy_id: int | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    items, total = scan_service.list_scans(session, current_user.id, strategy_id=strategy_id, limit=limit, offset=offset)
    return ScanListPage(items=[ScanSummary(**item) for item in items], total=total, limit=limit, offset=offset)


@router.get("/{scan_id}", response_model=ScanRead)
def get_scan_endpoint(
    scan_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    scan = scan_service.get_scan(session, scan_id, current_user.id)
    read = _read(scan)
    read.value_keys = scan_service.value_keys(session, scan_id)
    return read


@router.get("/{scan_id}/rows", response_model=ScanRowsPage)
def list_scan_rows_endpoint(
    scan_id: int,
    sort: str = Query(default="symbol", max_length=128, description="symbol, matched_rules, conditions_true, last_true, bar_ts, market.<field> or a values key (indicators.rsi14, price.close)"),
    descending: bool = False,
    rule: str | None = Query(default=None, max_length=255, description="Keep the rows where this rule is true"),
    matched_only: bool = Query(default=False, description="Keep the rows where at least one rule is true"),
    within: int | None = Query(default=None, ge=1, le=200, description="With rule or matched_only: true on one of the last N closed bars instead of on the last one"),
    symbol: str | None = Query(default=None, max_length=64),
    symbols: list[str] | None = Query(default=None, max_length=100, description="Exactly these symbols (repeat the parameter): the rows of the cards shown"),
    with_chart: bool = Query(default=False, description="Add to each row the bars of its miniature (chart): ask it for the page shown, not for a whole scan"),
    limit: int = Query(default=1000, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    scan, rows, total = scan_service.list_scan_rows(
        session, scan_id, current_user.id, sort=sort, descending=descending, rule=rule,
        matched_only=matched_only, within=within, symbol=symbol, symbols=symbols, limit=limit, offset=offset,
    )
    charts = scan_service.row_charts(session, rows) if with_chart else {}
    return ScanRowsPage(
        items=[
            # Field by field: reading ``chart`` off the row would load it, one query per row
            ScanRowRead(
                symbol=r.symbol, name=r.name, bar_ts=r.bar_ts, market=r.market, values=r.values, rules=r.rules,
                matched_rules=r.matched_rules, conditions_true=r.conditions_true,
                conditions_total=r.conditions_total, error=r.error, chart=charts.get(r.id),
            )
            for r in rows
        ],
        total=total, limit=limit, offset=offset,
        status=scan.status, symbols_total=scan.symbols_total, symbols_done=scan.symbols_done,
    )


@router.delete("/{scan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scan_endpoint(
    scan_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    scan_service.delete_scan(session, scan_id, current_user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
