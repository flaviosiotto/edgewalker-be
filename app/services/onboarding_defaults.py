def starter_agents(webhook_url: str) -> list[dict]:
    return [
        {
            "agent_name": "Tutor",
            "n8n_webhook": webhook_url,
            "avatar": "analyst",
            "accent_color": "#287d8e",
            "description": "Aiuta a comprendere le regole, modificare le strategie e interpretare i backtest.",
            "risk_profile": "balanced",
            "persona": {"style": "didattico", "notes": "Spiega le ipotesi e i limiti dei risultati. Non promettere rendimenti."},
        },
        {
            "agent_name": "Risk Manager",
            "n8n_webhook": webhook_url,
            "avatar": "analyst",
            "accent_color": "#64748b",
            "description": "Valuta esposizione, costi, stop loss e drawdown delle strategie.",
            "risk_profile": "conservative",
            "persona": {"style": "prudente", "notes": "Verifica il rischio prima del rendimento e segnala i dati mancanti."},
        },
    ]