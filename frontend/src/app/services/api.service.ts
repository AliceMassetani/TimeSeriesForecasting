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


  runBacktest(file: File, pidParams?: { kp?: number, ki?: number, kd?: number, exp?: number, scale_down?: number, quantile?: number, max_derivative?: number, acceleration_factor?: number }): Observable<any> {
    const formData = new FormData();
    formData.append('file', file); 
    if (pidParams) {
      if (this.isValidNum(pidParams.kp)) formData.append('pid_kp', String(pidParams.kp));
      if (this.isValidNum(pidParams.ki)) formData.append('pid_ki', String(pidParams.ki));
      if (this.isValidNum(pidParams.kd)) formData.append('pid_kd', String(pidParams.kd));
      if (this.isValidNum(pidParams.exp)) formData.append('pid_exp', String(pidParams.exp));
      if (this.isValidNum(pidParams.scale_down)) formData.append('pid_scale_down', String(pidParams.scale_down));
      if (this.isValidNum(pidParams.quantile)) formData.append('pid_quantile', String(pidParams.quantile));
      if (this.isValidNum(pidParams.max_derivative)) formData.append('pid_max_derivative', String(pidParams.max_derivative));
      if (this.isValidNum(pidParams.acceleration_factor)) formData.append('pid_acceleration_factor', String(pidParams.acceleration_factor));
    }
    return this.http.post(`${this.apiUrl}/backtest/run`, formData);
  }

  private isValidNum(val: any): boolean {
    return val !== null && val !== undefined && val !== '' && !isNaN(Number(val));
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
