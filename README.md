# TimeSeriesForecasting: Predictive Autoscaler ML

**Corso:** Progettazione di Applicazioni Web e Mobile  
**Università:** Università degli Studi di Camerino (UNICAM)

---

## 1. Panoramica del Progetto

Il progetto **TimeSeriesForecasting** è una piattaforma avanzata basata sul Machine Learning per l'analisi e la previsione di serie storiche, orientata all'ottimizzazione del *Predictive Autoscaling* di risorse cloud. Tramite l'impiego di modelli predittivi, il sistema mira ad anticipare i picchi di carico (capacità *InUse*) e a proporre un ridimensionamento proattivo delle risorse, minimizzando sprechi energetici ed economici.

L'applicazione è sviluppata come una moderna **Web App Full-Stack**. Il backend è un'API ad alte prestazioni basata su **FastAPI**, mentre il client è una **Single Page Application (SPA)** reattiva costruita con **Angular**. L'intero sistema è containerizzato per facilitare il deployment tramite Docker.

---

## 2. Architettura del Sistema e Scelte Progettuali

Il software è stato concepito applicando pattern architetturali moderni per garantire manutenibilità, separazione delle responsabilità e sicurezza.

### Layered Architecture (Backend)
Il backend adotta un'architettura a livelli (N-Tier) fortemente disaccoppiata:
- **Presentation Layer (Controllers / API Routers):** I moduli in `app/api/` (es. `auth.py`, `forecast.py`) si occupano esclusivamente del routing HTTP, della validazione dell'input e della formattazione dell'output.
- **Service Layer (`app/services/`):** Contiene la business logic pura dell'applicazione. Ad esempio, l'`AuthService` si occupa di generare JWT e validare credenziali, mantenendo i controller puliti.
- **Persistence / Data Access Layer (`app/repositories/`):** Implementa il *Repository Pattern* (es. `UserRepository`). Astrae la comunicazione con il database fornendo metodi standard (CRUD), rendendo il sistema agnostico rispetto all'ORM sottostante.
- **Entities & Schemas:** 
  - *Entities* (SQLAlchemy) in `app/entities/` mappano le tabelle del database relazionale.
  - *Schemas* (Pydantic) in `app/schemas/` gestiscono la serializzazione/deserializzazione e la validazione dei dati JSON scambiati via API.

### Autenticazione Stateless (JWT + BCrypt)
A differenza delle sessioni classiche *stateful* basate su cookie (che richiedono il salvataggio dello stato sul server o su Redis), l'applicativo implementa un'autenticazione **Stateless basata su JSON Web Tokens (JWT)**.
- **Vantaggi Architetturali:** Assoluta scalabilità orizzontale. Il server backend non deve ricordare quali utenti sono loggati; ogni richiesta contiene già nel token crittografato tutte le informazioni necessarie per l'identificazione.
- **Sicurezza:** Le password non vengono mai salvate in chiaro. Utilizziamo **BCrypt** per l'hashing sicuro: l'algoritmo applica un *salt* casuale a ogni password per difendersi efficacemente da attacchi *Rainbow Table* e inietta un fattore di costo algoritmico che rallenta gli attacchi di tipo *Brute Force*.

### Frontend Angular SPA
L'interfaccia utente è progettata per offrire un'esperienza fluida e premium:
- **Standalone Components:** Sviluppata seguendo il moderno approccio Angular (v15+), privo di NgModules. Ogni componente (es. `AuthComponent`) gestisce le proprie dipendenze.
- **Gestione dello Stato Reattivo (RxJS):** Lo stato dell'utente corrente (loggato/non loggato) è memorizzato e diffuso in tempo reale a tutta l'applicazione tramite un `BehaviorSubject` all'interno dell'`AuthService`.
- **Iniezione del Token (`AuthInterceptor`):** Tutte le chiamate HTTP (eccetto quelle per il login) vengono filtrate da un *Functional Interceptor* che aggancia automaticamente l'header `Authorization: Bearer <token>`, mantenendo il codice dei componenti pulito.
- **Protezione Rotte (`AuthGuard`):** Una *Functional Router Guard* analizza lo stato del servizio di autenticazione, impedendo l'accesso alle dashboard protette e scartando verso la pagina di login chiunque non possieda un JWT valido.

---

## 3. Tech Stack

### ⚙️ Backend
- **Core:** Python 3.10+, FastAPI
- **Database & ORM:** MariaDB, SQLAlchemy
- **Validazione:** Pydantic
- **Sicurezza:** PyJWT (Gestione Token), Passlib + BCrypt (Hashing Password)

### 🎨 Frontend
- **Framework:** Angular 15+ (TypeScript)
- **Gestione Stato:** RxJS
- **Interfaccia:** HTML5, Vanilla CSS3 (UI moderna con effetti *Glassmorphism* e animazioni fluide).

### 🚀 DevOps & ML
- **Containerization:** Docker, Docker Compose
- **Simulazione Cloud Locale:** LocalStack (AWS Lambda, EventBridge)
- **Modelli ML:** Forecast Models / PID Controllers.

---

## 4. Struttura delle API REST

L'API documenta automaticamente le sue rotte (disponibili via Swagger UI all'indirizzo `/docs`). Il sistema di rotte si divide in due macro-categorie:

### 🔓 Rotte Pubbliche (Nessuna Autenticazione Richiesta)

- `POST /api/auth/register`
  - **Body:** `{ "username": "...", "password": "..." }`
  - **Azione:** Crea un nuovo account, esegue l'hashing BCrypt della password e lo salva a DB.
  - **Risposte:** `201 Created` / `400 Bad Request` (Es. Username già in uso).

- `POST /api/auth/login`
  - **Body:** Form Data (`OAuth2PasswordRequestForm`) o JSON con username/password.
  - **Azione:** Valida le credenziali e restituisce il token.
  - **Risposte:** `200 OK` (Restituisce `{ "access_token": "...", "token_type": "bearer" }`) / `401 Unauthorized`.

### 🔒 Rotte Protette

Tutti i seguenti endpoint richiedono tassativamente un Token JWT valido passato negli headers HTTP della richiesta (`Authorization: Bearer <token>`). In caso contrario, il middleware FastAPI (Dipendenza `get_current_user`) restituirà un errore `401 Unauthorized` o `403 Forbidden`.

- `GET /forecast/*` (es. `/history`, `/run`)
  - **Azione:** Generazione predizioni future e interrogazione storico predittivo.
- `GET /backtest/*` (es. `/history`, `/run`)
  - **Azione:** Simulazione di affidabilità del modello su dati passati e relative metriche prestazionali (MSE, RMSE).
- `POST /train/*`
  - **Azione:** Addestramento e ricalibrazione del modello di Machine Learning con nuovi set di dati.