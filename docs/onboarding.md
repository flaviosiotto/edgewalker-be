# Onboarding Bitcoin e Forex

## Configurazione e deploy

1. Applicare `migrations/055_user_onboarding.sql` con il normale processo di
   migrazione, prima di avviare il nuovo backend. La colonna JSONB su `user`
   serve anche alle query degli utenti gia' esistenti.
   Applicare anche `migrations/056_drop_global_unique_connection_name.sql`:
   elimina l'unicita' globale residua su `connections.name` mantenendo quella
   per `(user_id, name)`. Senza questa migrazione, il secondo utente collide
   con i nomi delle connessioni di esempio e la registrazione viene annullata.
2. Gli agent standard sono definiti in `app/services/onboarding_defaults.py`:
   Tutor e Risk Manager, creati come record personali con nuovi ID. Tutor e'
   il manager iniziale delle strategie. Non serve `ONBOARDING_AGENT_IDS`.
   `ONBOARDING_AGENT_WEBHOOK_URL` permette di cambiare l'endpoint n8n dei nuovi
   agent; il default e' `/n8n/webhook/edgewalker-manager-v2`, dal workflow
   manager versionato nel repository devops. Pubblicare il workflow in n8n e
   configurare `N8N_INTERNAL_URL` per la risoluzione interna in Swarm.
3. Verificare che il workflow n8n sia adatto a utenti diversi:
   contesto e autorizzazione devono usare l'identita' ricevuta dal backend,
   senza credenziali personali o riferimenti fissi a un proprietario.
4. Configurare l'app OAuth cTrader e il redirect pubblico secondo la normale
   integrazione della piattaforma. L'onboarding usa `scope=accounts`.
5. Distribuire backend, frontend e immagine gateway aggiornata tramite Dokploy.
   Il gateway Binance deve supportare la modalita' senza credenziali senza
   eseguire chiamate private. Non cambiano i contratti Redis.

Il provisioning non interroga n8n o broker e non dipende da record agent
preesistenti. Un workflow non raggiungibile impedisce le chiamate AI, non la
creazione del workspace. I limiti e il credito AI del piano restano quelli
della piattaforma; l'onboarding non aggiunge credito o abbonamenti.

## Comportamento

La preparazione avviene nella stessa transazione della creazione dell'utente:
registrazione pubblica, accesso Google, creazione amministrativa e bootstrap
del primo amministratore usano `save_new_user()`. Se il provisioning fallisce,
viene annullata anche la creazione dell'utente. Email di verifica e approvazione
restano obbligatorie dove previste; le risorse iniziali non sbloccano l'accesso.

Per gli utenti gia' registrati, `/onboarding/prepare` conserva il recupero
idempotente al primo accesso, con lock sull'utente. In questo recupero gli
amministratori preesistenti e gli utenti con strategie o connessioni vengono
esclusi; chi ha workspace vuoto e stato `{}` riceve gli esempi senza configurare
ID template. Uno stato gia' preparato non viene sovrascritto.

Vengono creati:

- Tutor e Risk Manager personali e nuove chat, senza messaggi o cronologia;
- una connessione Binance Spot inattiva `data_only=true`, `read_only=true`, senza chiavi;
- un account tecnico Binance `spot`, tipo `data_only`, senza saldo;
- una strategia BTC/USDT oraria con size 0.001 BTC e chat di design;
- una connessione cTrader demo inattiva, senza token, con `read_only=true`;
- un account locale `provisional`, senza saldo o credenziali;
- una normale strategia EUR/USD oraria, con chat di design e regola di esempio.

Gli ID agent non sono condivisi tra utenti. Nome, profilo, persona e webhook
sono modificabili tramite le normali API agent con controllo del proprietario.
Il webhook puo' essere comune come servizio di esecuzione, ma non comporta
record agent, chat o configurazioni condivise. Modificare gli standard nel
codice non riscrive gli agent gia' creati. Nel recupero di utenti con agent
omonimi vengono mantenuti i loro record personali e le loro modifiche.

Il provisioning e' idempotente. Risorse eliminate successivamente non vengono
ricreate automaticamente. La guida puo' essere chiusa e ripresa solo dal menu
del profilo, alla voce Riprendi tour. Non viene creato un backtest prima del collegamento.

Entrambe le connessioni nascono inattive: nessun gateway viene avviato per un
utente appena registrato o ancora in attesa di verifica. Il primo accesso apre
l'elenco strategie con un tour di benvenuto ancorato agli elementi della pagina,
senza modale, sezioni aggiunte, scroll automatico o blocco della navigazione.
I suggerimenti sono brevi, esclusivamente floating, con sfondo distinto nei
temi chiaro e scuro. Presenta Bitcoin, Forex e gli agent
personali, poi lascia scegliere quale strategia esplorare. Il tour prosegue
nel workspace in fasi dedicate a regole, collegamento e primo backtest.
Il pulsante della guida Bitcoin attiva la connessione e collega i dati pubblici
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

Benvenuto, Bitcoin e Forex hanno progresso e chiusura della guida indipendenti,
salvati sul server. `PATCH /onboarding` accetta anche `track: welcome` e salva
`welcome_step` e `welcome_dismissed` nel JSONB esistente, senza nuove migrazioni.
Distribuire backend e frontend aggiornati insieme tramite Dokploy.
Il menu del profilo riprende Benvenuto, Bitcoin o Forex dal passo salvato.
Non ci sono pulsanti di ripresa nelle pagine, neppure in Help e Connessioni.
Un utente
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

Regressione PostgreSQL della migrazione 056 (solo tabella temporanea, nessuna
modifica ai dati applicativi; verifica anche la riesecuzione):

```bash
psql -X -v ON_ERROR_STOP=1 -f scripts/test_connection_name_migration.sql
```

Usare i normali parametri di connessione PostgreSQL dell'ambiente di test.

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