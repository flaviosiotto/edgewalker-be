# Deploy: riavvio backtest F0 "Riprova" + F1 timeout agente non fatale (03/11/2026)

Studio: `docs/valutazione-riavvio-backtest.md` (§3 e §4). F2 checkpoint e F3 Riprendi restano in sospeso.

## Cosa cambia

**F0 Riprova**
- BE: `POST /backtests/{id}/retry` (`strategy_service.retry_backtest`): su `completed`/`failed`/`error` crea un NUOVO backtest con gli stessi input, lo stesso `config` snapshot (non ri-snapshottato dalla strategia), lo stesso playbook d'origine ricopiato, `parameters` ereditati, nuova chat, `retried_from_backtest_id` = sorgente, e lo avvia. 409 su run non finito. Il sorgente resta intatto.
- BE: `strategy_backtests.failure_kind` (user_stop | shutdown | agent_timeout | runner_attach | runner | data | start | internal). `/stop` lo scrive `user_stop` PRIMA di chiedere il cancel al coordinator, che non lo sovrascrive; così lo stop manuale non è più "Cancelled by service shutdown".
- BE lista: `failure_kind`, `retried_from_backtest_id`, `agent_turns_timed_out` (proiezione scalare da `metrics`).
- Coordinator: ordini/cancel/close su un run finalizzato → 409 (bt 221: ordine eseguito 5 minuti dopo il failed e mai persistito); `failure_kind` classificato a `_finalize_run` (modello ORM del coordinator aggiornato con la colonna).
- FE: bottone "Riprova" in Backtests, nel pannello del backtest e nella card della strategia; nella lista "ripete #N" e conteggio turni scaduti; pannello Info con tipo di fallimento, turni scaduti e link al tentativo precedente. Il TaskModal segue l'avvio del nuovo run come per un backtest nuovo.
- MCP: `retry_backtest(backtest_id)`; docstring di `run_backtest` corretta (solo `pending`).

**F1 Timeout dell'agente non fatale**
- Coordinator: hold scaduto → rilascio + `expired_holds`/`agent_timeouts` nello stato, evento `agent_timeout` su `events:backtest-N`, `metrics.agent_turns_timed_out` e `metrics.agent_timeouts` a fine run. `parameters.on_agent_timeout = "fail"` ripristina il comportamento precedente (run fallito).
- Runner: `_finish_agent_hold` ritorna `ok | expired | timeout`; su `expired`/`timeout` (default `BACKTEST_ON_AGENT_TIMEOUT=continue`, env scritta dal BE da `parameters.on_agent_timeout`) aggiorna `agent_call.status = timeout`, pubblica l'evento runtime e scrive una riga di sistema nella chat. Con `fail` abortisce come prima.
- BE: `BacktestCreate.on_agent_timeout`; evento utente `backtest.agent_timeout` (severità warning) dal trigger su `agent_call` (migr. 073) → pagina Attività; la risposta tardiva dell'agente resta accettata e si applica alle barre successive (gli ordini non portano la correlazione del turno).
- FE: marker rombo ROSSO sul grafico per le chiamate con status `timeout`; voce nel catalogo Attività.

## Migrazione

`edgewalker-be/migrations/073_backtest_retry.sql`: due colonne + indice su `strategy_backtests`, trigger `trg_ew_webhook_agent_call`. Backup prima, dry run con `sed 's/^COMMIT;$/ROLLBACK;/'`, poi applicazione. **Prima del deploy del BE e del coordinator** (entrambi leggono `failure_kind`; il coordinator senza colonna nel DB fallirebbe la persist dei risultati).

## Ordine

1. migrazione 073
2. backend (immagine)
3. strategy-backtest (coordinator): nessun backtest in corso, i run in memoria vanno persi
4. strategy-runner (immagine; le live spawnate non sono toccate: il runner backtest è per-run)
5. frontend
6. mcp-svc

Nessuna variabile d'ambiente obbligatoria: i default sono `continue`.

## Verifica

- Backtest con rule `ask_agent` e `agent_timeout_s` basso (30 s) su un agente lento: il run deve arrivare a `completed`, il grafico mostrare il rombo rosso, la chat la riga "Turno dell'agente scaduto", Attività l'evento, la lista "1 turni dell'agente scaduti".
- Stop manuale: `failure_kind = user_stop`, messaggio "Stopped by user".
- Riprova su un run fallito: nuova riga con "ripete #N", il sorgente immutato, chat nuova.
- Ordine manuale dal workspace su un run finito: 409.

## Test

BE `tests/test_backtest_retry_db.py` (pgserver) + catalogo eventi (11 esiti); coordinator `tests/test_agent_timeout_hold.py` (con `PYTHONPATH=/home/flavio/playground/edgewalker` per il kit); vue-tsc verde. Il runner non ha una suite: compilato soltanto. `tests/test_lab_hub.py::test_start_creates_user_sets_auth_state_and_spawns` del BE fallisce anche senza queste modifiche (pre-esistente).
