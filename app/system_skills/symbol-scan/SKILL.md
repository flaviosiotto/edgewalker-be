---
name: symbol-scan
description: Scansiona il mercato della connessione con gli indicatori e le condizioni della strategia corrente: su quali simboli sono vere, sull'ultima barra chiusa.
---

# Scansionare i simboli

Usa questa procedura quando il trader chiede su quali altri simboli si presentano le condizioni della strategia, o dove ha senso applicarla. Una scansione legge gli indicatori del chart principale e le condizioni delle rule sull'ultima barra chiusa di molti simboli del mercato della connessione: produce una tabella, una riga per simbolo. Non classifica i simboli e non lancia ordini.

## Input

- timeframe (testo; facoltativo): timeframe delle barre, per esempio 1h; vuoto = quello del chart principale
- simboli (testo; facoltativo): simboli separati da virgola; vuoto = il mercato della connessione
- controvalore_minimo (numero; facoltativo): soglia di controvalore scambiato nelle 24 ore, in valuta di quotazione
- massimo_simboli (numero; default 100): quanti simboli al massimo, i più scambiati per primi
- regola (testo; facoltativo): la rule su cui concentrare la lettura; vuoto = tutte
- scansione (numero; facoltativo): id di una scansione già fatta da rileggere, senza lanciarne una nuova

## Procedura

1. Se è indicata una scansione già fatta, salta al punto 4 con quell'id.
2. La scansione usa la definizione SALVATA della strategia. Se in questo turno hai modificato la strategia senza salvarla, dillo prima di lanciare.
3. Lancia `run_scan` con gli input dati (i simboli come lista). Una sola scansione: non rilanciarla per cambiare ordinamento o filtro, quelli si chiedono a `get_scan`. Se dopo l'attesa non è conclusa, riferisci id e link e fermati: la leggerai quando il trader lo chiede. Se è fallita, riporta l'errore così com'è.
4. Leggi con `get_scan`. Prima la vista d'insieme: per ogni rule, su quanti simboli è vera, falsa, non valutabile. Poi, per la rule indicata (o per ciascuna rule che è vera su almeno un simbolo), una pagina di righe filtrata con `rule`. Non leggere la tabella intera: filtra e pagina.
5. Le condizioni che leggono `position.*`, `vars.*`, `studio.*` o un altro chart non sono valutabili in una scansione: dillo una volta, indicando quali, e non trattarle come false.
6. Se una rule non è vera su nessun simbolo, mostra quale condizione la blocca più spesso e di quanto: i due numeri confrontati sono nella riga.
7. Chiudi con cosa può fare il trader: aprire la tabella dal link per ordinarla e filtrarla, oppure copiare la strategia su uno dei simboli e lanciare un backtest massivo per vedere come si è comportata lì. Quali simboli provare lo decide lui.

## Forma della risposta

- Prima riga: quanti simboli letti, su quale timeframe e a quale barra (l'ora della barra chiusa), con il link alla scansione.
- Tabella "Rule | Vera | Falsa | Non valutabile", una riga per rule.
- Per la rule in esame, tabella dei simboli in cui è vera (massimo 10 righe): simbolo, controvalore 24h, e per ogni condizione il valore letto e quello con cui è confrontato. Se sono più di 10, dillo e rimanda al link.
- Niente premesse e niente riepilogo.

Regole: le righe non sono una classifica e una rule vera su una barra non è un segnale: non indicare mai "il simbolo migliore" né consigliare un ingresso. Riporta misure. Se ordini le righe, di' per quale chiave e che l'ordine è una scelta di lettura. Il controvalore sui derivati è stimato (volume per ultimo prezzo): dillo se lo usi come criterio. Mai più di una scansione per richiesta: ogni simbolo costa una lettura al broker.
