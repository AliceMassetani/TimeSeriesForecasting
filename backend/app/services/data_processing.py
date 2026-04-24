import pandas as pd
import io
from fastapi import HTTPException

class DataProcessingService:
    """
    Service dedicato alla pulizia, trasformazione e validazione dei dati (Principio SOLID: Single Responsibility).
    """

    def process_aws_csv(self, contents: bytes, target_column: str) -> pd.DataFrame:
        """
        Analizza, pulisce e valida un CSV proveniente da AWS (GRASP: Pure Fabrication).
        """
        # 1. Analisi Adattiva degli Header
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception:
            df = pd.DataFrame()

        # 2. Skip dinamico se necessario
        if df.empty or ("TimeStamp" not in df.columns and "Label" not in df.columns):
            raw_lines = contents.decode("utf-8", errors="ignore").splitlines()
            skip_rows = -1
            for i, line in enumerate(raw_lines[:50]):
                if "Label" in line or "TimeStamp" in line:
                    skip_rows = i
                    break
            
            if skip_rows != -1:
                df = pd.read_csv(io.BytesIO(contents), skiprows=skip_rows)
            else:
                raise HTTPException(status_code=400, detail="Impossibile identificare la riga degli header nel CSV.")

        # 3. Normalizzazione Colonne (Ridenominazione)
        if "Label" in df.columns and "TimeStamp" not in df.columns:
            df = df.rename(columns={"Label": "TimeStamp"})
        
        # 4. Filtro Colonne Indesiderate (SAAS_FLEET_DEV)
        dev_cols = [c for c in df.columns if "SAAS_FLEET_DEV" in c]
        if dev_cols:
            df = df.drop(columns=dev_cols)
        
        # 5. Pulizia Nomi (Rimuove prefissi PROD)
        df.columns = [c.replace("SAAS_FLEET_PROD ", "").strip() for c in df.columns]

        # 6. Validazione Esistenza Target
        # Mostra i nomi delle prime 10 colonne trovate
        if "TimeStamp" not in df.columns or target_column not in df.columns:
            available = ", ".join(df.columns[:10])
            raise HTTPException(
                status_code=400, 
                detail=f"Colonne mancanti ({target_column} o TimeStamp). Trovate: {available}..."
            )

        # 7. Parsing Temporale e Ordinamento
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        df = df.sort_values("TimeStamp")
        
        # 8. Valutazione Densità Adattiva (Resampling Implicito)
        if len(df) > 1:
            # Calcola la differenza media in minuti tra le righe
            avg_diff_min = df['TimeStamp'].diff().mean().total_seconds() / 60
            if avg_diff_min < 59:
                # Se i dati sono più densi dell'ora, prendiamo solo le ore in punto
                df = df[(df['TimeStamp'].dt.minute == 0) & (df['TimeStamp'].dt.second == 0)].copy()

        if df.empty:
            raise HTTPException(status_code=400, detail="Il set di dati risultante dopo la pulizia è vuoto.")

        return df

data_processing_service = DataProcessingService()
