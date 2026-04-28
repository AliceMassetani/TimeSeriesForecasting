import os
import pandas as pd
import torch
from darts import TimeSeries
from darts.models import TSMixerModel
from darts.utils.missing_values import fill_missing_values
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from fastapi import HTTPException

from .data_processing import data_processing_service

# Path dei modelli
MODEL_DIR = os.path.join("app", "ml_models")
CHAMPION_NAME = "tsmixer_champion.pt"
CANDIDATE_NAME = "tsmixer_candidate.pt"

CHAMPION_PATH = os.path.join(MODEL_DIR, CHAMPION_NAME)
CANDIDATE_PATH = os.path.join(MODEL_DIR, CANDIDATE_NAME)

class TrainingService:
    """
    Service responsabile per il retraining del modello TSMixer.
    """
    
    def __init__(self):
        # Parametri standard
        self.input_chunk_len = 168
        self.output_chunk_len = 8
        self.n_epochs = 50
        self.batch_size = 32
        self.hidden_size = 64
        self.ff_size = 64
        self.num_blocks = 2
        self.dropout = 0.1
        self.learning_rate = 1e-3

    def run_training_pipeline(self, contents: bytes, target: str) -> dict:
        """
        Esegue la pipeline completa: Pulizia -> Training -> Salvataggio.
        """
        try:
            # 1. Pulizia Dati (Riutilizzo del service esistente)
            df = data_processing_service.process_aws_csv(contents, target)
            
            if len(df) < self.input_chunk_len + self.output_chunk_len:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Dati insufficienti per il training. Necessari almeno {self.input_chunk_len + self.output_chunk_len} record."
                )

            # 2. Creazione TimeSeries
            series = fill_missing_values(
                TimeSeries.from_dataframe(df, time_col="TimeStamp", value_cols=target, freq="h")
            )

            # 3. Configurazione Modello (Parametri Standard)
            early_stop = EarlyStopping(
                monitor="train_loss",
                patience=10,
                min_delta=1e-4,
                mode="min",
            )

            model = TSMixerModel(
                input_chunk_length=self.input_chunk_len,
                output_chunk_length=self.output_chunk_len,
                hidden_size=self.hidden_size,
                ff_size=self.ff_size,
                num_blocks=self.num_blocks,
                dropout=self.dropout,
                likelihood=QuantileRegression(quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]),
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                optimizer_kwargs={"lr": self.learning_rate},
                pl_trainer_kwargs={
                    "enable_progress_bar": False,
                    "callbacks": [early_stop],
                    "accelerator": "auto" # Usa GPU se disponibile
                },
                random_state=42,
            )

            # 4. Addestramento
            print(f"Inizio addestramento TSMixer su {len(series)} record...")
            model.fit(series=series, verbose=False)

            # 5. Salvataggio (Come Candidato, non tocca la produzione)
            os.makedirs(MODEL_DIR, exist_ok=True)
            model.save(CANDIDATE_PATH)
            print(f"Modello candidato addestrato e salvato in {CANDIDATE_PATH}")

            return {
                "status": "success",
                "message": "Modello candidato addestrato con successo. Ora puoi caricarlo per testarlo o promuoverlo.",
                "records_trained": len(series)
            }

        except HTTPException as he:
            raise he
        except Exception as e:
            print(f"Errore durante il training: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Errore interno durante il training: {str(e)}")

    def promote_candidate(self) -> dict:
        """
        Promuove il modello candidato a Champion (Produzione) con backup.
        """
        if not os.path.exists(CANDIDATE_PATH):
            raise HTTPException(status_code=404, detail="Nessun modello candidato trovato da promuovere.")
        
        try:
            import shutil
            # 1. Crea un backup del champion attuale se esiste
            if os.path.exists(CHAMPION_PATH):
                BACKUP_PATH = CHAMPION_PATH + ".bak"
                shutil.copy(CHAMPION_PATH, BACKUP_PATH)
                print(f"Backup creato: {BACKUP_PATH}")

            # 2. Copia candidato -> champion
            shutil.copy(CANDIDATE_PATH, CHAMPION_PATH)
            return {
                "status": "success",
                "message": "Modello candidato promosso a Champion. Il vecchio champion è stato salvato come backup."
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Errore durante la promozione: {str(e)}")

    def rollback_champion(self) -> dict:
        """
        Ripristina l'ultimo modello Champion dal backup.
        """
        BACKUP_PATH = CHAMPION_PATH + ".bak"
        if not os.path.exists(BACKUP_PATH):
            raise HTTPException(status_code=404, detail="Nessun backup trovato per il rollback.")
        
        try:
            import shutil
            # Ripristina backup -> champion
            # Il champion attuale viene sovrascritto, ma è salvato in tsmixer_candidate
            shutil.copy(BACKUP_PATH, CHAMPION_PATH)
            return {
                "status": "success",
                "message": "Rollback completato. Il modello di produzione è stato ripristinato dal backup."
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Errore durante il rollback: {str(e)}")

training_service = TrainingService()
