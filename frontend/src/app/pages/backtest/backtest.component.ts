import { Component, ElementRef, OnInit, signal, viewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { Chart } from 'chart.js/auto';
import { DecimalPipe } from '@angular/common';

interface BacktestMetrics {
  rmse: number;
  mse: number;
  mae: number;
  r2: number;
  rmse_pid?: number;
  mse_pid?: number;
  mae_pid?: number;
  r2_pid?: number;

  // Nuove metriche
  under_count?: number;
  over_count?: number;
  under_sum?: number;
  over_sum?: number;
  under_count_pid?: number;
  over_count_pid?: number;
  under_sum_pid?: number;
  over_sum_pid?: number;
}

//Risposta di esecuzione backtest
interface BacktestResponse {
  message: string;
  results_saved: number;
  metrics: BacktestMetrics;
  status: string;
}

//Dati per grafico backtest
interface BacktestChartData {
  labels: string[];
  actual: number[];
  prediction: number[];
  prediction_p10: number[];
  prediction_p90: number[];
  diff_instances: number[];
  prediction_rounded: number[];
  prediction_p10_rounded: number[];
  prediction_p90_rounded: number[];
  actual_rounded: number[];
  diff_rounded_instances: number[];
  prediction_pid?: number[];
  prediction_pid_rounded?: number[];
  metrics?: BacktestMetrics;
}

@Component({
  selector: 'app-backtest',
  standalone: true,
  imports: [FormsModule, DecimalPipe],
  template: `
    <div class="page-container">
      <div class="page-header">
        <h2>Analisi Storica (Backtest)</h2>
        <p class="subtitle">Carica un file CSV storico per validare le prestazioni del modello sui dati passati.</p>
      </div>
    
      <!-- 1. ESECUZIONE NUOVO BACKTEST -->
      <div class="upload-section">
        <h3>1. Esegui Nuovo Backtest</h3>
        <p class="subtitle mb-1-5">Avvia un nuovo test caricando un file di dati.</p>
        
        <div class="flex-row flex-gap-1-5">
          <label class="file-upload">
            <input type="file" (change)="onFileSelected($event)" accept=".csv" style="display: none;">
            <span class="upload-btn">Sfoglia CSV</span>
          </label>
          <button 
            class="primary-btn" 
            (click)="runBacktest()" 
            [disabled]="!selectedFile || isBacktesting()">
            {{ isBacktesting() ? 'Esecuzione...' : 'Avvia Backtest' }}
          </button>
          @if (selectedFile) {
            <span class="subtitle" style="margin-top: 0;">File: {{ selectedFile.name }}</span>
          }
        </div>

        <!-- CONFIGURAZIONE PID -->
        <details class="mt-2">
          <summary class="text-primary" style="cursor: pointer; font-weight: 600; font-size: 0.9rem;">
            Configurazione PID (Autoscaler)
          </summary>
          <div class="filters-row mt-1" style="background: rgba(255,255,255,0.02); padding: 1.5rem; border-radius: 8px; flex-wrap: wrap;">
            <div class="filter-group">
              <label>Quantile Setpoint</label>
              <select [(ngModel)]="pidQuantile" class="sleek-field">
                <option [ngValue]="0.5">P50 (mediana)</option>
                <option [ngValue]="0.75">P75</option>
                <option [ngValue]="0.9">P90 (default)</option>
                <option [ngValue]="0.95">P95</option>
              </select>
            </div>
            <div class="filter-group">
              <label>Kp (Proporzionale)</label>
              <input type="number" [(ngModel)]="pidKp" step="0.05" min="0" max="5" placeholder="0.3 (default)">
            </div>
            <div class="filter-group">
              <label>Ki (Integrale)</label>
              <input type="number" [(ngModel)]="pidKi" step="0.005" min="0" max="1" placeholder="0.0 (default)">
            </div>
            <div class="filter-group">
              <label>Kd (Derivata)</label>
              <input type="number" [(ngModel)]="pidKd" step="0.01" min="0" max="2" placeholder="0.05 (default)">
            </div>
            <div class="filter-group">
              <label>Exp Derivata</label>
              <input type="number" [(ngModel)]="pidExp" step="0.1" min="1" max="3" placeholder="1.0 (default)">
            </div>
            <div class="filter-group">
              <label>Scale Down Penalty</label>
              <input type="number" [(ngModel)]="pidScaleDown" step="0.1" min="0" max="1" placeholder="0.6 (default)">
            </div>
            <div class="filter-group">
              <label class="has-tooltip">
                Max Derivative (Cap) <span class="info-icon">i</span>
                <span class="tooltip-text">Se vuoto, il sistema analizza la volatilità storica (P99 delle variazioni) per impostare un tetto automatico.</span>
              </label>
              <input type="number" [(ngModel)]="pidMaxDerivative" step="10" min="1" placeholder="Auto-calc">
            </div>
            <div class="filter-group">
              <label class="has-tooltip">
                Acceleration Factor <span class="info-icon">i</span>
                <span class="tooltip-text">Moltiplica la spinta del PID durante i picchi improvvisi. Default 1.2.</span>
              </label>
              <input type="number" [(ngModel)]="pidAccelerationFactor" step="0.1" min="1" max="5" placeholder="1.2 (default)">
            </div>
          </div>
          <div class="mt-1" style="background: rgba(96, 165, 250, 0.05); border-left: 3px solid var(--primary-blue); border-radius: 4px; margin-left: 1.5rem; padding: 0.75rem 1.2rem;">
            <p style="font-size: 0.75rem; color: var(--text-muted); margin: 0;">
              <strong style="color: var(--primary-blue);"> Logica Adattiva:</strong> 
              Se i parametri "Max Derivative" e "Safety Limit" sono vuoti, il sistema analizza il dataset caricato. 
              Viene calcolata la <strong>volatilità (P99 delle variazioni)</strong> per impostare un tetto di sicurezza che permetta reazioni rapide 
              ma protegga da errori o rumore nei dati.
            </p>
          </div>
        </details>

        @if (isBacktesting()) {
          <div class="flex-row mt-1-5 text-primary">
            <div class="spinner"></div>
            <span style="font-weight: 500;">Esecuzione Backtest in corso... L'operazione può richiedere tempo per file grandi.</span>
          </div>
        }

        <!-- RISULTATO BACKTEST (SUCCESSO) -->
        @if (backtestResult()) {
          <div class="alert-box success" style="margin-top: 1.5rem; display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem;">
            <div style="display: flex; align-items: center; gap: 1rem;">
              <span class="alert-icon" style="margin: 0; position: static; font-size: 1.2rem;">✓</span>
              <div class="alert-content">
                <strong style="margin: 0;">Backtest Completato!</strong>
                <p style="margin: 0; opacity: 0.8;">{{ backtestResult()?.message }}</p>
              </div>
            </div>
            <div class="flex-row flex-gap-1">
              <button class="secondary-btn" (click)="exportToCsv('original')" style="background: rgba(153, 102, 255, 0.1); border: 1px solid var(--primary-blue); color: #fff; font-size: 0.75rem;">
                Esporta Originale
              </button>
              <button class="secondary-btn" (click)="exportToCsv('pid')" style="background: rgba(45, 212, 191, 0.1); border: 1px solid #2dd4bf; color: #fff; font-size: 0.75rem;">
                Esporta PID
              </button>
            </div>
          </div>
        }

        <!-- AVVISO DATI TAGLIATI -->
        @if (dataCapped()) {
          <div class="alert-box warning" style="margin-top: 1.5rem;">
            <span class="alert-icon">⚠️</span>
            <div class="alert-content">
              <strong>Visualizzazione limitata</strong>
              <p>
                Il file contiene {{ originalPointsCount() }} punti. Per garantire la fluidità del browser, 
                stiamo mostrando solo gli ultimi 1.400 punti (circa 2 mesi di dati).
              </p>
            </div>
          </div>
        }

        <!-- ERRORE BACKTEST -->
        @if (backtestError()) {
          <div class="alert-box error" style="margin-top: 1.5rem;">
            <span class="alert-icon">✕</span>
            <div class="alert-content">
              <strong>Errore durante il Backtest</strong>
              <p>{{ backtestError() }}</p>
            </div>
          </div>
        }
      </div>

      <!-- 2. VISUALIZZA STORICO -->
      <div class="upload-section">
        <h3>2. Visualizza Risultati Salvati</h3>
        <p class="subtitle mb-1-5">Recupera i dati di un test già effettuato.</p>
        
        <div class="filters-row">
          <div class="filter-group">
            <label>Punti</label>
            <input type="number" [(ngModel)]="limit" placeholder="Es. 50">
          </div>
          <div class="filter-group">
            <label>Inizio</label>
            <input type="datetime-local" [(ngModel)]="startDate">
          </div>
          <div class="filter-group">
            <label>Fine</label>
            <input type="datetime-local" [(ngModel)]="endDate">
          </div>
          <button class="primary-btn" (click)="loadBacktestChart()" [disabled]="isLoadingChart()">
            {{ isLoadingChart() ? 'Caricamento...' : 'Vedi grafico' }}
          </button>
        </div>      

        <!-- ERRORE CARICAMENTO STORICO -->
        @if (loadError()) {
          <div class="alert-box error" style="margin-top: 1.5rem;">
            <span class="alert-icon">✕</span>
            <div class="alert-content">
              <p>{{ loadError() }}</p>
            </div>
          </div>
        }
      </div>

      <!-- RISULTATI -->
      @if (hasData()) {
        <!-- SEZIONE ORIGINALE -->
        <div class="page-header" style="margin-top: 2rem;">
          <h3>Risultati Modello Originale (TSMixer)</h3>
          <p class="subtitle">Dati grezzi di previsione senza correzioni.</p>
        </div>

        <div class="legend-custom">
          <div class="legend-item"><span class="dot actual"></span> Reale</div>
          <div class="legend-item"><span class="dot p50"></span> Previsione (P50)</div>
          <div class="legend-item"><span class="dot p90"></span> Area di Previsione (P10-P90)</div>
          <div class="legend-item"><span class="dot diff"></span> Differenza</div>
        </div>

        <div class="chart-container">
          <div class="chart-scroll-container">
            <div class="chart-wrapper" [style.width]="getChartWidth()">
              <canvas #backtestChart></canvas>
            </div>
          </div>
        </div>

        <!-- Metriche Originali -->
        <div class="metrics-grid mt-1-5">
          <div class="metrics-card">
            <span class="label">RMSE</span>
            <span class="value">{{ metrics()?.rmse | number: '1.2-2' }}</span>
          </div>
          <div class="metrics-card">
            <span class="label">MAE</span>
            <span class="value">{{ metrics()?.mae | number: '1.2-2' }}</span>
          </div>
          <div class="metrics-card">
            <span class="label">MSE</span>
            <span class="value">{{ metrics()?.mse | number: '1.2-2' }}</span>
          </div>
          <div class="metrics-card">
            <span class="label">R²</span>
            <span class="value">{{ metrics()?.r2 | number: '1.4-4' }}</span>
          </div>
        </div>

        <div class="metrics-grid mt-1">
          <div class="metrics-card border-danger">
            <span class="label">Sotto-dimensionamento (Volte)</span>
            <span class="value">{{ metrics()?.under_count }}</span>
          </div>
          <div class="metrics-card border-danger">
            <span class="label">Sotto-dimensionamento (Numero di instanze)</span>
            <span class="value">{{ metrics()?.under_sum | number: '1.0-0' }}</span>
          </div>
          <div class="metrics-card">
            <span class="label">Sovra-dimensionamento (Volte)</span>
            <span class="value">{{ metrics()?.over_count }}</span>
          </div>
          <div class="metrics-card">
            <span class="label">Sovra-dimensionamento (Numero di instanze)</span>
            <span class="value">{{ metrics()?.over_sum | number: '1.0-0' }}</span>
          </div>
        </div>

        @if (chartData?.prediction_pid_rounded) {
          <div class="divider"></div>

          <div class="page-header">
            <h3 class="text-pid">Correzione PID (Non-lineare)</h3>
            <p class="subtitle">Piano di allocazione ottimizzato con logica PID derivativa per eliminare il lag.</p>
          </div>

          <div class="legend-custom" style="margin-top: 1rem;">
            <div class="legend-item"><span class="dot actual"></span> Reale</div>
            <div class="legend-item"><span class="dot p50" style="background: rgba(153, 102, 255, 1)"></span> Allocazione PID</div>
            <div class="legend-item"><span class="dot diff" style="background: rgba(255, 159, 64, 1)"></span> Differenza (PID - Reale)</div>
            <div class="legend-item"><span class="dot p90" style="background: rgba(153, 102, 255, 0.3)"></span> Banda Previsione</div>
          </div>

          <div class="chart-container">
            <div class="chart-scroll-container">
              <div class="chart-wrapper" [style.width]="getChartWidth()">
                <canvas #pidChart></canvas>
              </div>
            </div>
          </div>

          <!-- Metriche PID -->
          <div class="metrics-grid mt-1-5">
            <div class="metrics-card pid-accent">
              <span class="label">RMSE (PID)</span>
              <div class="value-row">
                <span class="value">{{ metrics()?.rmse_pid | number: '1.2-2' }}</span>
                <span class="comparison-badge" [class.improvement]="(metrics()?.rmse_pid || 0) < (metrics()?.rmse || 0)" [class.worsening]="(metrics()?.rmse_pid || 0) > (metrics()?.rmse || 0)">
                  {{ calculateVariation(metrics()?.rmse_pid, metrics()?.rmse) }}
                </span>
              </div>
            </div>
            <div class="metrics-card pid-accent">
              <span class="label">MAE (PID)</span>
              <div class="value-row">
                <span class="value">{{ metrics()?.mae_pid | number: '1.2-2' }}</span>
                <span class="comparison-badge" [class.improvement]="(metrics()?.mae_pid || 0) < (metrics()?.mae || 0)" [class.worsening]="(metrics()?.mae_pid || 0) > (metrics()?.mae || 0)">
                  {{ calculateVariation(metrics()?.mae_pid, metrics()?.mae) }}
                </span>
              </div>
            </div>
            <div class="metrics-card pid-accent">
              <span class="label">MSE (PID)</span>
              <div class="value-row">
                <span class="value">{{ metrics()?.mse_pid | number: '1.2-2' }}</span>
                <span class="comparison-badge" [class.improvement]="(metrics()?.mse_pid || 0) < (metrics()?.mse || 0)" [class.worsening]="(metrics()?.mse_pid || 0) > (metrics()?.mse || 0)">
                  {{ calculateVariation(metrics()?.mse_pid, metrics()?.mse) }}
                </span>
              </div>
            </div>
            <div class="metrics-card pid-accent">
              <span class="label">R² (PID)</span>
              <div class="value-row">
                <span class="value">{{ metrics()?.r2_pid | number: '1.4-4' }}</span>
                <span class="comparison-badge" [class.improvement]="(metrics()?.r2_pid || 0) > (metrics()?.r2 || 0)" [class.worsening]="(metrics()?.r2_pid || 0) < (metrics()?.r2 || 0)">
                  {{ calculateVariation(metrics()?.r2_pid, metrics()?.r2) }}
                </span>
              </div>
            </div>
          </div>

          <div class="metrics-grid mt-1">
            <div class="metrics-card pid-accent">
              <span class="label">Sotto-dimensionamento (volte) (PID)</span>
              <div class="value-row">
                <span class="value">{{ metrics()?.under_count_pid }}</span>
                <span class="comparison-badge" [class.improvement]="(metrics()?.under_count_pid || 0) < (metrics()?.under_count || 0)" [class.worsening]="(metrics()?.under_count_pid || 0) > (metrics()?.under_count || 0)">
                  {{ calculateVariation(metrics()?.under_count_pid, metrics()?.under_count) }}
                </span>
              </div>
            </div>
            <div class="metrics-card pid-accent">
              <span class="label">Sotto-dimensioamento (Numero di instanze) (PID)</span>
              <div class="value-row">
                <span class="value">{{ metrics()?.under_sum_pid | number: '1.0-0' }}</span>
                <span class="comparison-badge" [class.improvement]="(metrics()?.under_sum_pid || 0) < (metrics()?.under_sum || 0)" [class.worsening]="(metrics()?.under_sum_pid || 0) > (metrics()?.under_sum || 0)">
                  {{ calculateVariation(metrics()?.under_sum_pid, metrics()?.under_sum) }}
                </span>
              </div>
            </div>
            <div class="metrics-card pid-accent">
              <span class="label">Sovra-dimensionamento (volte) (PID)</span>
              <div class="value-row">
                <span class="value">{{ metrics()?.over_count_pid }}</span>
                <span class="comparison-badge" [class.improvement]="(metrics()?.over_count_pid || 0) < (metrics()?.over_count || 0)" [class.worsening]="(metrics()?.over_count_pid || 0) > (metrics()?.over_count || 0)">
                  {{ calculateVariation(metrics()?.over_count_pid, metrics()?.over_count) }}
                </span>
              </div>
            </div>
            <div class="metrics-card pid-accent">
              <span class="label">Sovra-dimensionamento (Numero di instanze) (PID)</span>
              <div class="value-row">
                <span class="value">{{ metrics()?.over_sum_pid | number: '1.0-0' }}</span>
                <span class="comparison-badge" [class.improvement]="(metrics()?.over_sum_pid || 0) < (metrics()?.over_sum || 0)" [class.worsening]="(metrics()?.over_sum_pid || 0) > (metrics()?.over_sum || 0)">
                  {{ calculateVariation(metrics()?.over_sum_pid, metrics()?.over_sum) }}
                </span>
              </div>
            </div>
          </div>
        }
      }
    </div>
  `,
  styles: [`
    .filter-group {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      min-width: 140px;
    }
    .has-tooltip {
      position: relative;
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: help;
    }
    .info-icon {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 14px;
      height: 14px;
      border: 1px solid var(--primary-blue);
      border-radius: 50%;
      font-size: 0.65rem;
      font-weight: bold;
      color: var(--primary-blue);
      font-style: italic;
    }
    .tooltip-text {
      visibility: hidden;
      width: 200px;
      background-color: #1a1f2e;
      color: #fff;
      text-align: center;
      border-radius: 6px;
      padding: 8px;
      position: absolute;
      z-index: 100;
      bottom: 125%;
      left: 0;
      opacity: 0;
      transition: opacity 0.3s;
      font-size: 0.7rem;
      text-transform: none;
      letter-spacing: normal;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
      border: 1px solid rgba(255,255,255,0.1);
      pointer-events: none;
    }
    .has-tooltip:hover .tooltip-text {
      visibility: visible;
      opacity: 1;
    }
    .filter-group input, .filter-group select {
      height: 38px;
    }
  `]
})
export class BacktestComponent implements OnInit {
  selectedFile: File | null = null;
  limit?: number;
  startDate?: string;
  endDate?: string;

  // PID configurazione utente
  pidKp?: number;
  pidKi?: number;
  pidKd?: number;
  pidExp?: number;
  pidScaleDown?: number;
  pidMaxDerivative?: number;
  pidAccelerationFactor?: number;
  pidQuantile: number = 0.9;

  isBacktesting = signal(false);
  isLoadingChart = signal(false);
  hasData = signal(false);
  metrics = signal<BacktestMetrics | null>(null);
  backtestResult = signal<BacktestResponse | null>(null);
  backtestError = signal<string | null>(null);
  dataCapped = signal(false);
  originalPointsCount = signal(0);
  loadError = signal<string | null>(null);
  chartData: BacktestChartData | null = null;

  calculateVariation(newVal: number | undefined, oldVal: number | undefined): string {
    if (newVal === undefined || oldVal === undefined || oldVal === 0) return '';
    const delta = ((newVal - oldVal) / Math.abs(oldVal)) * 100;
    const sign = delta >= 0 ? '+' : '';
    return `${sign}${delta.toFixed(1)}%`;
  }

  getChartWidth() {
    if (!this.chartData || !this.chartData.labels) return '100%';
    const points = this.chartData.labels.length;
    return `${points * 22}px`;
  }

  chartCanvas = viewChild<ElementRef<HTMLCanvasElement>>('backtestChart');
  pidCanvas = viewChild<ElementRef<HTMLCanvasElement>>('pidChart');
  chart: any;
  pidChart: any;

  constructor(private apiService: ApiService) { }

  ngOnInit() {
    this.loadBacktestChart();
  }

  onFileSelected(event: any) {
    this.selectedFile = event.target.files?.[0] || null;
  }

  runBacktest() {
    if (!this.selectedFile) return;

    console.log('--- AVVIO BACKTEST ---');
    this.isBacktesting.set(true);
    this.backtestResult.set(null);
    this.backtestError.set(null);
    this.dataCapped.set(false);

    this.apiService.runBacktest(this.selectedFile, {
      kp: this.pidKp, ki: this.pidKi, kd: this.pidKd,
      exp: this.pidExp, scale_down: this.pidScaleDown,
      quantile: this.pidQuantile,
      max_derivative: this.pidMaxDerivative,
      acceleration_factor: this.pidAccelerationFactor
    }).subscribe({
      next: (response: BacktestResponse) => {
        console.log('--- BACKTEST COMPLETATO ---', response);
        this.backtestResult.set(response);
        this.metrics.set(response.metrics);
        this.loadBacktestChart();
        this.isBacktesting.set(false);
      },
      error: (err: any) => {
        console.error('--- ERRORE BACKTEST ---', err);
        this.isBacktesting.set(false);
        const detail = err.error?.detail || err.message || 'Errore durante l\'esecuzione del backtest.';
        this.backtestError.set(detail);
      }
    });
  }

  exportToCsv(mode: string = 'all') {
    this.apiService.exportBacktestCsv(mode).subscribe({
      next: (blob: Blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `backtest_${mode}_${new Date().getTime()}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      },
      error: (err) => {
        console.error('Errore durante l\'esportazione:', err);
      }
    });
  }

  loadBacktestChart() {
    console.log('--- CARICAMENTO STORICO BACKTEST ---');
    this.isLoadingChart.set(true);
    this.loadError.set(null);
    this.dataCapped.set(false);

    this.apiService.getBacktestHistory(this.limit, this.startDate, this.endDate).subscribe({
      next: (data: BacktestChartData) => {
        console.log('--- STORICO CARICATO ---', data);
        this.metrics.set(data.metrics || null);
        this.createChart(data);
        this.isLoadingChart.set(false);
      },
      error: (err: any) => {
        console.error('--- ERRORE CARICAMENTO STORICO ---', err);
        this.isLoadingChart.set(false);
        const detail = err.error?.detail || 'Errore nel recupero dello storico backtest.';
        this.loadError.set(detail);
      }
    });
  }

  private createChart(data: BacktestChartData) {
    // --- LIMITE TECNICO CANVAS (STESSO DEL FORECAST) ---
    const MAX_POINTS = 1400;
    this.dataCapped.set(false);

    let displayData = { ...data };
    if (data.labels && data.labels.length > MAX_POINTS) {
      this.dataCapped.set(true);
      this.originalPointsCount.set(data.labels.length);
      const startIdx = data.labels.length - MAX_POINTS;

      displayData.labels = data.labels.slice(startIdx);
      displayData.prediction_p10_rounded = data.prediction_p10_rounded.slice(startIdx);
      displayData.prediction_p90_rounded = data.prediction_p90_rounded.slice(startIdx);
      displayData.prediction_rounded = data.prediction_rounded.slice(startIdx);
      displayData.diff_rounded_instances = data.diff_rounded_instances.slice(startIdx);
      displayData.actual_rounded = data.actual_rounded.slice(startIdx);
      if (data.prediction_pid_rounded) {
        displayData.prediction_pid_rounded = data.prediction_pid_rounded.slice(startIdx);
      }
    }

    this.chartData = displayData;
    this.hasData.set(true);
    const canvas = this.chartCanvas()?.nativeElement;
    const canvasPid = this.pidCanvas()?.nativeElement;
    if (!canvas) return;

    const context = canvas.getContext('2d');
    if (!context) return;

    if (this.chart) this.chart.destroy();
    if (this.pidChart) this.pidChart.destroy();

    this.chart = new Chart(context, {
      type: 'line',
      data: {
        labels: displayData.labels,
        datasets: [
          {
            label: 'P10',
            data: displayData.prediction_p10_rounded,
            borderColor: 'rgba(255, 99, 132, 0)',
            pointRadius: 0,
            fill: false,
            tension: 0.4
          },
          {
            label: 'Prediction Band (P10-P90)',
            data: displayData.prediction_p90_rounded,
            borderColor: 'rgba(153, 102, 255, 0.5)',
            backgroundColor: 'rgba(153,102,255,0.3)',
            fill: 0,
            tension: 0.4,
            borderWidth: 0,
            cubicInterpolationMode: 'monotone',
            pointRadius: 0
          },
          {
            label: 'Prediction (P50)',
            data: displayData.prediction_rounded,
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0)',
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 2,
            borderWidth: 2
          },
          {
            label: 'Diff',
            data: displayData.diff_rounded_instances,
            borderColor: 'rgba(255, 159, 64, 1)',
            backgroundColor: 'rgba(255, 159, 64, 0.1)',
            fill: true,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 2
          },
          {
            label: 'Actual',
            data: displayData.actual_rounded,
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 4
          }
        ]
      },
      options: {
        devicePixelRatio: window.devicePixelRatio || 2,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'Analisi Backtest', color: '#ffffff' }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: { color: '#cccccc' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            title: {
              display: true,
              text: 'Capacità (Istanze)',
              color: '#ffffff',
              font: { size: 16, weight: 'bold', family: 'Inter' },
              padding: { bottom: 20 }
            }
          },
          x: {
            ticks: { color: '#cccccc', maxRotation: 45, minRotation: 45 },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            title: {
              display: true,
              text: 'Tempo',
              color: '#ffffff',
              font: { size: 16, weight: 'bold', family: 'Inter' },
              padding: { top: 20 }
            }
          }
        }
      }
    });

    // 2. Chart PID (Aggiunta Nuova)
    if (canvasPid && displayData.prediction_pid_rounded) {
      const ctxPid = canvasPid.getContext('2d');
      if (ctxPid) {
        const diffPid = displayData.prediction_pid_rounded.map((val, i) => val - (displayData.actual_rounded[i] || 0));

        this.pidChart = new Chart(ctxPid, {
          type: 'line',
          data: {
            labels: displayData.labels,
            datasets: [
              {
                label: 'P10',
                data: displayData.prediction_p10_rounded,
                borderColor: 'rgba(255, 99, 132, 0)',
                pointRadius: 0,
                fill: false,
                tension: 0.4
              },
              {
                label: 'Prediction Band (P10-P90)',
                data: displayData.prediction_p90_rounded,
                borderColor: 'rgba(153, 102, 255, 0.5)',
                backgroundColor: 'rgba(153,102,255,0.3)',
                fill: 0,
                tension: 0.4,
                borderWidth: 0,
                cubicInterpolationMode: 'monotone',
                pointRadius: 0
              },
              {
                label: 'Prediction (PID)',
                data: displayData.prediction_pid_rounded,
                borderColor: 'rgba(153, 102, 255, 1)',
                backgroundColor: 'rgba(153, 102, 255, 0)',
                fill: false,
                tension: 0.4,
                cubicInterpolationMode: 'monotone',
                pointRadius: 2,
                borderWidth: 2
              },
              {
                label: 'Diff (PID)',
                data: diffPid,
                borderColor: 'rgba(255, 159, 64, 1)',
                backgroundColor: 'rgba(255, 159, 64, 0.1)',
                fill: true,
                tension: 0.4,
                cubicInterpolationMode: 'monotone',
                pointRadius: 2
              },
              {
                label: 'Actual',
                data: displayData.actual_rounded,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: false,
                tension: 0.4,
                cubicInterpolationMode: 'monotone',
                pointRadius: 4
              }
            ]
          },
          options: {
            devicePixelRatio: window.devicePixelRatio || 2,
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
              title: { display: true, text: 'Analisi Backtest (Correzione PID)', color: '#ffffff' }
            },
            scales: {
              y: {
                beginAtZero: true,
                ticks: { color: '#cccccc' },
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                title: {
                  display: true,
                  text: 'Capacità (Istanze)',
                  color: '#ffffff',
                  font: { size: 16, weight: 'bold', family: 'Inter' },
                  padding: { bottom: 20 }
                }
              },
              x: {
                ticks: { color: '#cccccc', maxRotation: 45, minRotation: 45 },
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                title: {
                  display: true,
                  text: 'Tempo',
                  color: '#ffffff',
                  font: { size: 16, weight: 'bold', family: 'Inter' },
                  padding: { top: 20 }
                }
              }
            }
          }
        });
      }
    }
  }
}
