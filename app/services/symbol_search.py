"""
Symbol Search Service - Search symbols across data sources.

Provides unified symbol search for Yahoo Finance and IBKR.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


class SymbolSearchError(Exception):
    """Error searching for symbols."""
    pass


def search_symbols(
    query: str,
    source: str,
    asset_type: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Search for symbols from a data source.
    
    Args:
        query: Search query (symbol pattern or company name)
        source: Data source ('yahoo' or 'ibkr')
        asset_type: Filter by asset type ('stock', 'futures', 'index', 'etf')
        limit: Maximum number of results
    
    Returns:
        List of symbol info dicts with: symbol, name, type, exchange, etc.
    """
    source = source.lower()
    
    if source == "yahoo":
        return _search_yahoo(query, asset_type, limit)
    elif source == "ibkr":
        return _search_ibkr(query, asset_type, limit)
    else:
        raise SymbolSearchError(f"Unknown data source: {source}")


def _search_yahoo(
    query: str,
    asset_type: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Search Yahoo Finance for symbols.
    
    Uses the Yahoo Finance autosuggest API.
    """
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            "q": query,
            "quotesCount": limit,
            "newsCount": 0,
            "listsCount": 0,
            "enableFuzzyQuery": True,
            "quotesQueryId": "tss_match_phrase_query",
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        quotes = data.get("quotes", [])
        results = []
        
        # Map Yahoo type codes to our asset types
        yahoo_type_map = {
            "EQUITY": "stock",
            "ETF": "etf",
            "INDEX": "index",
            "FUTURE": "futures",
            "OPTION": "option",
            "MUTUALFUND": "fund",
            "CURRENCY": "currency",
            "CRYPTOCURRENCY": "crypto",
        }
        
        for quote in quotes:
            quote_type = quote.get("quoteType", "")
            mapped_type = yahoo_type_map.get(quote_type, quote_type.lower())
            
            # Filter by asset type if specified
            if asset_type and mapped_type != asset_type:
                continue
            
            symbol_info = {
                "symbol": quote.get("symbol", ""),
                "name": quote.get("shortname") or quote.get("longname", ""),
                "type": mapped_type,
                "exchange": quote.get("exchange", ""),
                "exchange_display": quote.get("exchDisp", ""),
                "score": quote.get("score", 0),
                "source": "yahoo",
            }
            results.append(symbol_info)
        
        return results[:limit]
        
    except requests.RequestException as e:
        logger.error(f"Yahoo symbol search failed: {e}")
        raise SymbolSearchError(f"Yahoo Finance search failed: {e}")
    except Exception as e:
        logger.exception(f"Error searching Yahoo for '{query}'")
        raise SymbolSearchError(f"Symbol search error: {e}")


def _search_ibkr(
    query: str,
    asset_type: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Search IBKR for symbols.
    
    Requires active IBKR connection. Falls back to cached contracts.
    """
    try:
        # Try to connect to IBKR via the edgewalker library
        import sys
        import os
        
        # Add edgewalker to path if not already there
        edgewalker_path = os.environ.get("EDGEWALKER_PATH", "/home/flavio/playground/edgewalker")
        if edgewalker_path not in sys.path:
            sys.path.insert(0, edgewalker_path)
        
        from edgewalker.marketdata.sources.ibkr import IBKRConfig
        from ib_async import IB
        
        # Load IBKR config
        import yaml
        ibkr_config_path = os.path.join(edgewalker_path, "configs", "ibkr.yaml")
        
        ibkr_cfg = IBKRConfig()
        if os.path.exists(ibkr_config_path):
            with open(ibkr_config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
                conn = cfg.get("connection", {})
                ibkr_cfg = IBKRConfig(
                    host=conn.get("host", ibkr_cfg.host),
                    port=conn.get("port", ibkr_cfg.port),
                    client_id=conn.get("client_id", ibkr_cfg.client_id + 10),  # Offset to avoid conflicts
                )
        
        # Map asset types to IBKR security types
        sec_type_map = {
            "stock": "STK",
            "futures": "FUT",
            "index": "IND",
            "etf": "STK",  # ETFs are stocks in IBKR
            "option": "OPT",
        }
        
        sec_type = sec_type_map.get(asset_type) if asset_type else None
        
        results = []
        
        try:
            ib = IB()
            ib.connect(ibkr_cfg.host, ibkr_cfg.port, clientId=ibkr_cfg.client_id)
            
            # Request matching symbols
            matches = ib.reqMatchingSymbols(query)
            
            for match in matches[:limit]:
                contract = match.contract
                contract_type = getattr(contract, "secType", "")
                
                # Map IBKR type back to our type
                mapped_type = {v: k for k, v in sec_type_map.items()}.get(contract_type, contract_type.lower())
                
                # Filter by asset type if specified
                if sec_type and contract_type != sec_type:
                    continue
                
                symbol_info = {
                    "symbol": getattr(contract, "symbol", ""),
                    "name": getattr(match, "companyName", "") or getattr(contract, "localSymbol", ""),
                    "type": mapped_type,
                    "exchange": getattr(contract, "primaryExchange", "") or getattr(contract, "exchange", ""),
                    "currency": getattr(contract, "currency", "USD"),
                    "con_id": getattr(contract, "conId", None),
                    "local_symbol": getattr(contract, "localSymbol", ""),
                    "source": "ibkr",
                }
                
                # For futures, add expiry info
                if contract_type == "FUT":
                    symbol_info["expiry"] = getattr(contract, "lastTradeDateOrContractMonth", "")
                
                results.append(symbol_info)
            
            ib.disconnect()
            return results
            
        except Exception as conn_err:
            logger.warning(f"IBKR connection failed: {conn_err}. Trying cached contracts...")
            
            # Fall back to cached contracts if available
            return _search_ibkr_cache(query, asset_type, limit, edgewalker_path)
            
    except ImportError as e:
        logger.error(f"IBKR modules not available: {e}")
        raise SymbolSearchError(
            "IBKR search requires edgewalker library. "
            "Make sure EDGEWALKER_PATH is set correctly."
        )
    except Exception as e:
        logger.exception(f"Error searching IBKR for '{query}'")
        raise SymbolSearchError(f"IBKR search error: {e}")


def _search_ibkr_cache(
    query: str,
    asset_type: Optional[str],
    limit: int,
    edgewalker_path: str,
) -> list[dict[str, Any]]:
    """Search IBKR cached contract files.
    
    Used as fallback when IBKR connection is not available.
    """
    import json
    import os
    from pathlib import Path
    
    cache_dir = Path(edgewalker_path) / "artifacts" / "ibkr_cache"
    
    if not cache_dir.exists():
        return []
    
    results = []
    query_lower = query.lower()
    
    for cache_file in cache_dir.glob("contracts_*.json"):
        try:
            with open(cache_file, "r") as f:
                contracts = json.load(f)
            
            for contract in contracts:
                symbol = contract.get("localSymbol", "") or ""
                
                # Match query against symbol
                if query_lower in symbol.lower():
                    symbol_info = {
                        "symbol": contract.get("localSymbol", ""),
                        "name": contract.get("localSymbol", ""),
                        "type": "futures",  # Cached contracts are typically futures
                        "exchange": contract.get("exchange", ""),
                        "currency": contract.get("currency", "USD"),
                        "con_id": contract.get("conId"),
                        "expiry": contract.get("lastTradeDateOrContractMonth", ""),
                        "source": "ibkr_cache",
                    }
                    results.append(symbol_info)
                    
                    if len(results) >= limit:
                        return results
                        
        except Exception as e:
            logger.debug(f"Error reading cache file {cache_file}: {e}")
            continue
    
    return results
