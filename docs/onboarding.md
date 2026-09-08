# Onboarding Bitcoin e Forex

## Configurazione e deploy

1. Applicare `migrations/055_user_onboarding.sql` con il normale processo di
   migrazione, prima di avviare il nuovo backend. La colonna JSONB su `user`
   serve anche alle query degli utenti gia' esistenti.
2. Impostare sul backend `ONBOARDING_AGENT_IDS` come array JSON, per esempio
   `[1,2]`, con gli ID degli agent approvati dall'amministratore. Non sono
   selezionati automaticamente agent appartenenti ad altri utenti. Il primo
   template fornisce il manager delle due strategie iniziali.
3. Verificare che i webhook n8n dei template siano adatti a utenti diversi:
   contesto e autorizzazione devono usare l'identita' ricevuta dal backend,
   senza credenziali personali o riferimenti fissi al proprietario del template.
4. Configurare l'app OAuth cTrader e il redirect pubblico secondo la normale
   integrazione della piattaforma. L'onboarding usa `scope=accounts`.
5. Distribuire backend, frontend e immagine gateway aggiornata tramite Dokploy.
   Il gateway Binance deve supportare la modalita' senza credenziali senza
   eseguire chiamate private. Non cambiano i contratti Redis.

Se gli ID sono assenti, inesistenti o senza webhook, `/onboarding/prepare`
risponde 503 senza creare risorse parziali. Il frontend mostra un errore con
Riprova. I limiti e il credito AI del piano restano quelli della piattaforma;
l'onboarding non aggiunge credito o abbonamenti.

## Comportamento

La preparazione avviene al primo accesso autenticato, non durante la
registrazione. Gli amministratori e gli utenti che hanno gia' una strategia
o una connessione vengono esclusi automaticamente.

In una sola transazione, con lock sull'utente, vengono creati:

- copie dei template agent e nuove chat, senza messaggi o cronologia;
- una connessione Binance Spot `data_only=true`, `read_only=true`, senza chiavi;
- un account tecnico Binance `spot`, tipo `data_only`, senza saldo;
- una strategia BTC/USDT oraria con size 0.001 BTC e chat di design;
- una connessione cTrader demo inattiva, senza token, con `read_only=true`;
- un account locale `provisional`, senza saldo o credenziali;
- una normale strategia EUR/USD oraria, con chat di design e regola di esempio.

Il provisioning e' idempotente. Risorse eliminate successivamente non vengono
ricreate automaticamente. La guida puo' essere chiusa e ripresa dal workspace
o dalla pagina Help. Non viene creato un backtest prima del collegamento.

Il primo accesso apre Bitcoin. Il pulsante della guida collega i dati pubblici
usando la normale API delle connessioni: non servono conto Binance, API key o
depositi. Grafici e backtest usano il feed Spot BTC/USDT. La disponibilita'
dipende dall'accesso alle API pubbliche Binance dal server, dai limiti di
richiesta e dalla disponibilita' dello storico, non da dati simulati forniti
dalla piattaforma. L'account tecnico puo' essere usato anche per altre
strategie e backtest, ma non contribuisce ai totali di performance account.

Il backend scarta le chiavi nel mapping ambiente di una connessione data-only;
il gateway senza credenziali non interroga saldi, ordini o posizioni private e
rifiuta inserimenti e cancellazioni di ordini. Il Live viene rifiutato anche
dal backend. Per trading autenticato si usa una connessione distinta.

Bitcoin e Forex hanno progresso e chiusura della guida indipendenti. Un utente
con onboarding Forex gia' preparato riceve il percorso Bitcoin mancante al
successivo accesso, se la strategia Forex con il suo manager esiste ancora.
Le risorse Bitcoin eliminate dopo il provisioning non vengono ricreate.

Il conto provvisorio non compare nei conti di performance. Non puo' essere
usato per nuove strategie, copie, backtest o live. Il design resta modificabile,
ma non monta il grafico e non richiede storico o precisione al gateway.

L'utente deve possedere un conto demo presso un broker cTrader: il cTrader ID
e' solo l'identita' di accesso, non un conto broker. Dopo OAuth, deve salvare e
connettere la connessione, quindi tornare alla guida e scegliere un conto
connesso e uno strumento Forex verificato nel catalogo del gateway.

L'attivazione ricollega la stessa strategia al conto reale scoperto, aggiorna
simbolo e metadati del primo grafico, conserva le regole salvate dall'utente e
rimuove il conto provvisorio quando non piu' referenziato da strategie. I
grafici secondari eventualmente aggiunti dall'utente non vengono riscritti.
Una connessione iniziale abbandonata a favore di un'altra non viene eliminata.

La size dell'esempio Forex e' 1000 unita'. Il catalogo corrente non espone minimo e
incremento dei volumi: vanno verificati sul conto del broker. Non e' una
raccomandazione di investimento. La simulazione usa capitale e costi scelti
nel backtest, non il saldo broker.

La connessione iniziale resta in sola lettura anche dopo l'attivazione e la
validazione live la rifiuta. Per il trading serve una connessione operativa
autorizzata separatamente con scope trading; questa guida non promuove la
connessione iniziale a operativa.

## Verifiche

Test unitari senza database o gateway reali (Session mock):

```bash
PYTHONPATH=.:../edgewalker venv/bin/python -m unittest discover -s scripts -p test_onboarding.py -v
```

Verifica frontend dalla directory `edgewalker-fe`:

```bash
npx vue-tsc --noEmit -p tsconfig.app.json
```

Test gateway dalla directory `edgewalker-runtime`, con un interprete che abbia
le dipendenze del gateway:

```bash
python scripts/test_binance_data_only.py -v
```

Dopo il deploy verificare con un nuovo utente: provisioning senza duplicati
anche da due tab, connessione pubblica Binance senza chiavi, grafico e backtest
Bitcoin, chiusura e ripresa di entrambe le guide, annullamento e ritorno OAuth,
scoperta account, selezione EUR/USD, conservazione delle regole modificate,
caricamento storico e primo backtest. Verificare che gli utenti gia' attivi
non ricevano nuove risorse. I test locali con mock non sostituiscono queste
verifiche, ne' validano migrazione, lock PostgreSQL o autorizzazione provider.