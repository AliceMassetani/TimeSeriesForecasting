import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ForecastChart, BacktestChart, TrainingResult } from '../models/api-data.model';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  // L'indirizzo del tuo backend FastAPI
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) { }

  /**
   * Recupera i dati del grafico per il Forecast futuro
   */
  getForecastHistory(): Observable<ForecastChart> {
    return this.http.get<ForecastChart>(`${this.apiUrl}/forecast/history`);
  }

  /**
   * Recupera lo storico dei Backtest formattato per il grafico
   */
  getBacktestHistory(): Observable<BacktestChart> {
    return this.http.get<BacktestChart>(`${this.apiUrl}/backtest/history`);
  }

  /**
   * Avvia l'addestramento caricando un file CSV
   */
  trainModel(file: File): Observable<TrainingResult> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<TrainingResult>(`${this.apiUrl}/train/`, formData);
  }

  /**
   * Carica il modello candidato per il test
   */
  loadCandidate(): Observable<any> {
    return this.http.post(`${this.apiUrl}/train/load-candidate`, {});
  }

  /**
   * Ripristina il modello champion
   */
  loadChampion(): Observable<any> {
    return this.http.post(`${this.apiUrl}/train/load-champion`, {});
  }

  /**
   * Promuove il candidato a champion
   */
  promoteModel(): Observable<any> {
    return this.http.post(`${this.apiUrl}/train/promote`, {});
  }

  /**
   * Rollback all'ultimo champion funzionante
   */
  rollbackModel(): Observable<any> {
    return this.http.post(`${this.apiUrl}/train/rollback`, {});
  }
}
