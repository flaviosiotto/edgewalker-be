---
name: backtest-review
description: Valuta un backtest concluso: diagnosi dalle misure, confronto con un altro run, modifiche proposte o applicate alla strategia.
---

# Valutare un backtest

Usa questa procedura quando il trader chiede di valutare, spiegare, confrontare o migliorare un backtest.

## Input

- backtest (backtest; facoltativo): il run da valutare; vuoto = l'ultimo completato della strategia
- confronto (backtest; facoltativo): un altro run con cui confrontarlo
- obiettivo (scelta: più trade | meno drawdown | stabilità nel tempo | parametri; facoltativo): cosa vuoi migliorare
- modifiche (scelta: proponi | applica e rilancia; default proponi): cosa fare delle modifiche

## Procedura

1. Leggi il run con `get_backtest` (se i suoi dati sono già nel contesto del turno, non rileggerlo). Se lo stato non è `completed`, riferisci lo stato e fermati.
2. Diagnosi dalle misure già calcolate: le 3-5 che spiegano il risultato (rendimento, drawdown massimo, profit factor, win rate, numero di trade; poi distribuzione per motivo di uscita, episodi di drawdown, esposizione, segmenti temporali, concentrazione sui trade migliori). Non chiedere mai il ledger grezzo.
3. `get_backtest_trades` solo se un'ipotesi richiede i singoli trade (ore di ingresso, durata, motivo di uscita). `get_backtest_trials` solo se il run ha un'esplorazione dei parametri.
4. Con un run di confronto: cosa è cambiato fra i due (definizione, periodo, modalità) e cosa spiega la differenza. Due run in modalità diverse (replay e massiva) non si confrontano barra per barra: dillo.
5. Proposta: al massimo 3 modifiche alla definizione, ognuna con la misura che dovrebbe muovere e in che verso. Se c'è un obiettivo, le modifiche servono quello.
6. Cosa fare delle modifiche:
   - `proponi`: non salvare nulla. Se le alternative sono discrete, chiedi al trader quale applicare e prosegui nello stesso turno con la sua risposta.
   - `applica e rilancia`: `get_strategy`, poi `update_strategy` con le modifiche; poi rilancia il backtest sullo stesso periodo e nella stessa modalità del run valutato e confronta le misure nuove con le vecchie.
7. Il playbook prodotto dal run (`list_agent_lessons` del backtest): se merita la promozione dillo, la decisione resta al trader.

## Forma della risposta

- Prima riga: l'esito in una frase con i due numeri che contano.
- Tabella "Misura | Valore | Lettura", massimo 5 righe.
- Modifiche numerate, una riga ciascuna.
- Niente premesse e niente riepilogo: l'analisi estesa resta nel ragionamento.

Regole: niente verdetti del tipo "la strategia funziona": descrivi cosa mostrano le misure e su quale base statistica (numero di trade, stabilità nel tempo). Niente promozione del playbook senza richiesta esplicita. Confronta sempre con il periodo e i costi di commissione usati dal run.
