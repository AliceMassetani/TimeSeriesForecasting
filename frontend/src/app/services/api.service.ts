import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
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
  getBacktestHistory(limit?: number, startDate?: string, endDate?: string): Observable<BacktestChart> {
    let params = new HttpParams();
    if (limit) params = params.set('limit', limit.toString());
    if (startDate) params = params.set('start_date', startDate);
    if (endDate) params = params.set('end_date', endDate);
    return this.http.get<BacktestChart>(`${this.apiUrl}/backtest/history`, { params });
  }


  runBacktest(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.apiUrl}/backtest/run`, formData);
  }

  /**
   * Avvia la generazione del forecast caricando un file CSV
   */
  runForecast(file: File, historyHours: number = -1): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.apiUrl}/forecast/run?history_hours=${historyHours}`, formData);
  }

  /**
   * Avvia l'addestramento caricando un file CSV e parametri opzionali
   */
  trainModel(file: File, params?: any): Observable<TrainingResult> {
    const formData = new FormData();
    formData.append('file', file);
    
    if (params) {
      Object.keys(params).forEach(key => {
        if (params[key] !== null && params[key] !== undefined) {
          formData.append(key, params[key].toString());
        }
      });
    }
    
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
