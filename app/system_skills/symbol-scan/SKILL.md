---
name: symbol-scan
description: Scansiona il mercato di una connessione con le rule di un template di strategia, con un filtro libero su indicatori classici, oppure con le rule della strategia corrente: su quali simboli sono vere adesso o lo sono state nelle ultime barre chiuse.
---

# Scansionare i simboli

Usa questa procedura quando il trader cerca un mercato: su quali simboli si presentano le condizioni di un template, uno screening su indicatori classici (RSI, medie, volume), oppure dove altro si presentano le condizioni della strategia corrente. Lo scanner sta a monte della strategia: dal simbolo scelto si crea poi la strategia dal template. Una scansione legge gli indicatori del chart principale e le condizioni delle rule sulle ultime barre chiuse di molti simboli del mercato della connessione: produce una tabella, una riga per simbolo. I valori sono dell'ultima barra chiusa; per ogni rule e condizione la riga dice anche quando è stata vera nella finestra guardata, così compare anche una condizione scattata e subito rientrata (un incrocio). Non classifica i simboli e non lancia ordini.

## Input

- template (testo; facoltativo): nome o id del template di strategia da scansionare; vuoto = filtro libero, oppure le rule della strategia corrente se la chat ne ha una
- connessione (testo; facoltativo): la connessione di cui leggere il mercato; vuoto = quella della strategia corrente, altrimenti chiedila
- mercati_fissi (testo; facoltativo): per template o strategie con più chart, slot da leggere su un simbolo fisso, per esempio "mkt=BTC-USDT-SWAP"; gli altri seguono il simbolo della riga
- timeframe (testo; facoltativo): solo per il filtro libero, per esempio 1h; i timeframe di template e strategie non si cambiano
- simboli (testo; facoltativo): simboli separati da virgola; vuoto = il mercato della connessione
- tipi (testo; facoltativo): tipi di strumento a cui restringere il mercato, separati da virgola (forex, index, commodity, stock, crypto, etf, futures, cfd); utile sulle connessioni cTrader; vuoto = tutti
- controvalore_minimo (numero; facoltativo): soglia di controvalore scambiato nelle 24 ore, in valuta di quotazione; non vale sulle connessioni senza dati a 24 ore (cTrader)
- massimo_simboli (numero; default 100): quanti simboli al massimo, i più scambiati per primi
- barre_indietro (numero; default 20): quante barre chiuse guardare indietro, l'ultima compresa; 1 = solo l'ultima
- parametri (testo; facoltativo): valori dei params della strategia da usare solo per questa scansione, per esempio "rsi_entry=25, max_bars=10"; vuoto = quelli salvati
- regola (testo; facoltativo): la rule su cui concentrare la lettura; vuoto = tutte
- filtro (testo; facoltativo): filtro libero al posto delle rule della strategia, per esempio "RSI 14 sotto 30" o "chiusura sopra EMA 200"
- scansione (numero; facoltativo): id di una scansione già fatta da rileggere, senza lanciarne una nuova

## Procedura

1. Se è indicata una scansione già fatta, salta al punto 4 con quell'id.
2. Scegli la sorgente, una sola. Template: trova l'id con `list_strategy_templates` e passa `template_id` e `connection_id` (se la connessione non è chiara, chiedila). Filtro libero: vedi il punto 3. Strategia corrente: nessun template e nessun filtro; la scansione usa la definizione SALVATA, letta come template (il simbolo della strategia non conta): se in questo turno hai modificato la strategia senza salvarla, dillo prima di lanciare.
3. Lancia `run_scan` con gli input dati (i simboli come lista, barre_indietro come `lookback`, parametri come oggetto `params` nome → numero; la strategia non viene modificata). Se è dato un filtro libero, traducilo in `indicators` e `conditions` di `run_scan` (tipo e parametri degli indicatori da `list_indicators`; campi `price.*` e `indicators.<nome>`): la scansione allora non usa né template né rule della strategia, e il timeframe va indicato (se manca chiedilo, oppure nella chat di una strategia usa quello del suo chart principale e dillo). Riporta in una riga come hai tradotto il filtro. Una sola scansione: non rilanciarla per cambiare ordinamento o filtro, quelli si chiedono a `get_scan`. Se dopo l'attesa non è conclusa, riferisci id e link e fermati: la leggerai quando il trader lo chiede. Se è fallita, riporta l'errore così com'è.
4. Leggi con `get_scan`. Prima la vista d'insieme: per ogni rule, su quanti simboli è vera, falsa, non valutabile. Poi, per la rule indicata (o per ciascuna rule vera su almeno un simbolo nella finestra), una pagina di righe filtrata con `rule`, `within` uguale alle barre guardate e `sort` `last_true`: così leggi anche i simboli su cui è scattata ed è rientrata. Non leggere la tabella intera: filtra e pagina.
5. Le condizioni che leggono `position.*`, `vars.*`, `studio.*` non sono valutabili in una scansione: dillo una volta, indicando quali, e non trattarle come false. Gli altri chart della strategia (altri timeframe, simboli di contesto) vengono letti: se su un simbolo mancano le barre di un chart, le sue condizioni risultano non valutabili su quella riga.
6. Rispetta il ruolo delle rule (`role`). Le rule di ingresso (buy, sell), quelle che chiamano l'agente e il filtro libero sono ciò che la scansione cerca. Le rule di uscita (close) no: in una scansione non c'è una posizione aperta, quindi non possono scattare. Riportale a parte, e solo come informazione ("su questi simboli la condizione di uscita è già vera").
7. Se una rule non è vera su nessun simbolo, mostra quale condizione la blocca più spesso e di quanto: i due numeri confrontati sono nella riga.
8. Chiudi con cosa può fare il trader: aprire la tabella dal link per ordinarla e filtrarla, oppure creare la strategia dal template su uno dei simboli (`create_strategy_from_template`, dopo aver chiesto il conto) e lanciare un backtest massivo per vedere come si è comportata lì. Quali simboli provare lo decide lui.

## Forma della risposta

- Prima riga: quanti simboli letti, su quale timeframe e a quale barra (l'ora della barra chiusa), con il link alla scansione.
- Se la strategia ha params: una riga con i valori usati, segnando quelli cambiati per questa scansione.
- Tabella "Rule | Vera adesso | Vera nelle ultime N barre | Falsa adesso | Non valutabile", una riga per rule di ingresso; le rule di uscita in una riga a parte sotto la tabella.
- Per la rule in esame, tabella dei simboli in cui è vera o lo è stata nella finestra (massimo 10 righe): simbolo, quando ("adesso, da N barre" oppure "N barre fa"), controvalore 24h, e per ogni condizione il valore letto e quello con cui è confrontato. Se sono più di 10, dillo e rimanda al link.
- Niente premesse e niente riepilogo.

Regole: le righe non sono una classifica e una rule vera su una barra non è un segnale: non indicare mai "il simbolo migliore" né consigliare un ingresso. Riporta misure. Una rule scattata N barre fa non è vera adesso: dillo così, senza presentarla come attuale. Se ordini le righe, di' per quale chiave e che l'ordine è una scelta di lettura. Il controvalore sui derivati è stimato (volume per ultimo prezzo): dillo se lo usi come criterio. Mai più di una scansione per richiesta: ogni simbolo costa una lettura al broker.
