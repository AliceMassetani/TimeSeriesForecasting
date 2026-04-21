# Usa un'immagine base ufficiale di Python con build tools
FROM python:3.12

# Installa le dipendenze di sistema necessarie per MariaDB
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libmariadb-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Imposta la directory di lavoro nel container
WORKDIR /app

# Copia il file requirements.txt nella directory di lavoro
COPY requirements.txt .

# Installa PyTorch CPU-only (saves ~1GB) then all dependencies
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice sorgente nel container
COPY . .

# Espone la porta su cui girerà FastAPI (default 8000)
EXPOSE 8000

# Comando per avviare l'applicazione
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]