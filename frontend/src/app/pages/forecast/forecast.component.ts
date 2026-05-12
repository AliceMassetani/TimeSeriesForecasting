import { Component, OnInit, ElementRef, signal, viewChild, computed } from '@angular/core';
import { ApiService } from '../../services/api.service';
import { ForecastChart } from '../../models/api-data.model';
import { Chart } from 'chart.js/auto';

import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-forecast',
  standalone: true,
  imports: [FormsModule],
  template: `
    <div class="page-container">
      <div class="page-header">
        <h2>Previsione Live (Forecast)</h2>
        <p class="subtitle">Capacità In-Use prevista per le prossime ore rispetto ai dati reali recenti.</p>
      </div>

      <!-- SEZIONE UPLOAD -->
      <div class="upload-section">
        <h3>Genera Nuove Previsioni</h3>
        <p class="subtitle mb-1-5">Carica l'ultimo export dei dati per proiettare la capacità futura.</p>
        
        <div class="flex-row flex-gap-1-5">
          <label class="file-upload">
            <input type="file" (change)="onFileSelected($event)" accept=".csv" style="display: none;">
            <span class="upload-btn">Sfoglia CSV</span>
          </label>

          <select #rangeSelect (change)="historyWindow.set(parseRange(rangeSelect.value))" class="sleek-field">
            <option value="168">1 Week</option>
            <option value="336">2 Weeks</option>
            <option value="720">1 Month</option>
            <option value="-1" selected>All Data</option>
          </select>

          <select [(ngModel)]="forecastQuantile" class="sleek-field">
            <option [ngValue]="0.5">P50 (mediana)</option>
            <option [ngValue]="0.75">P75</option>
            <option [ngValue]="0.9">P90 (default)</option>
            <option [ngValue]="0.95">P95</option>
          </select>

          <button 
            class="primary-btn" 
            (click)="runForecast()" 
            [disabled]="!selectedFile() || isForecasting()">
            {{ isForecasting() ? 'Generazione...' : 'Aggiorna Previsioni' }}
          </button>
          
          @if (selectedFile()) {
            <span class="subtitle" style="margin-top: 0;">File: {{ selectedFile()?.name }}</span>
          }
        </div>

        <!-- ERRORE GENERAZIONE FORECAST -->
        @if (forecastError()) {
          <div class="alert-box error" style="margin-top: 1.5rem;">
            <span class="alert-icon">✕</span>
            <div class="alert-content">
              <strong>Errore nella generazione del Forecast</strong>
              <p>{{ forecastError() }}</p>
            </div>
          </div>
        }
      </div>

      @if (loadError()) {
        <div class="alert-box error">
          <span class="alert-icon">✕</span>
          <div class="alert-content">
            <p>{{ loadError() }}</p>
          </div>
        </div>
      }

      @if (dataCapped()) {
        <div class="alert-box warning">
          <span class="alert-icon">⚠️</span>
          <div class="alert-content">
            <strong>Visualizzazione limitata</strong>
            <p>Mostrando solo gli ultimi 1.400 punti (circa 2 mesi di dati).</p>
          </div>
        </div>
      }

      @if (hasData()) {
        <!-- GRAFICO 1: ORIGINALE -->
        <div class="page-header" style="margin-top: 2rem;">
          <h3>Risultati Modello Originale (TSMixer)</h3>
          <p class="subtitle">Dati grezzi di previsione senza correzioni.</p>
        </div>

        <div class="legend-custom">
          <div class="legend-item"><span class="dot actual"></span> Reale</div>
          <div class="legend-item"><span class="dot prediction"></span> Previsione</div>
          <div class="legend-item"><span class="dot p90"></span> Area di Previsione (P10-P90)</div>
        </div>

        <div class="chart-container">
          <div class="chart-scroll-container">
            <div class="chart-wrapper" [style.width]="chartWidth()" [style.height.px]="chartHeight()">
              <canvas #forecastChart></canvas>
            </div>
          </div>
        </div>

        <!-- GRAFICO 2: PID -->
        @if (chartData()?.prediction_pid_rounded) {
          <div class="divider"></div>

          <div class="page-header">
            <h3 class="text-pid">Allocazione PID (Controller Non-Lineare)</h3>
            <p class="subtitle">Piano di allocazione corretto tramite il controller PID Non-Lineare.</p>
          </div>

          <div class="legend-custom" style="margin-top: 1rem;">
            <div class="legend-item"><span class="dot actual"></span> Reale</div>
            <div class="legend-item"><span class="dot prediction"></span> Allocazione PID</div>
            <div class="legend-item"><span class="dot p90" style="background: rgba(153, 102, 255, 0.3)"></span> Banda Previsione Originale</div>
          </div>

          <div class="chart-container">
            <div class="chart-scroll-container">
              <div class="chart-wrapper" [style.width]="chartWidth()" [style.height.px]="chartHeight()">
                <canvas #pidForecastChart></canvas>
              </div>
            </div>
          </div>
        }

        <!-- GRAFICO 3: KALMAN -->
        @if (chartData()?.prediction_kalman_rounded) {
          <div class="divider"></div>

          <div class="page-header">
            <h3 class="text-kalman">Allocazione Kalman (Correzione Bias Adattiva)</h3>
            <p class="subtitle">Piano di allocazione ottimizzato tramite filtro di Kalman per la rimozione del bias.</p>
          </div>

          <div class="legend-custom" style="margin-top: 1rem;">
            <div class="legend-item"><span class="dot actual"></span> Reale</div>
            <div class="legend-item"><span class="dot prediction"></span> Allocazione Kalman</div>
            <div class="legend-item"><span class="dot p90" style="background: rgba(153, 102, 255, 0.3)"></span> Banda Previsione Originale</div>
          </div>

          <div class="chart-container">
            <div class="chart-scroll-container">
              <div class="chart-wrapper" [style.width]="chartWidth()" [style.height.px]="chartHeight()">
                <canvas #kalmanForecastChart></canvas>
              </div>
            </div>
          </div>
        }
      } @else if (isLoading()) {
        <div class="flex-row" style="height: 100%; justify-content: center; margin-top: 4rem;">
           <div class="spinner"></div>
           <span class="text-primary">Caricamento grafico...</span>
        </div>
      }
    </div>
  `,
  styles: [`
    .chart-wrapper {
      --chart-height: 400px;
    }
    .text-kalman { color: rgba(251, 191, 36, 1); }
  `]
})
export class ForecastComponent implements OnInit {
  isLoading = signal(true);
  isForecasting = signal(false);
  hasData = signal(false);
  chartHeight = signal(400);
  selectedFile = signal<File | null>(null);
  historyWindow = signal(-1);
  forecastQuantile = 0.9;
  forecastError = signal<string | null>(null);
  loadError = signal<string | null>(null);
  dataCapped = signal(false);
  originalPointsCount = signal(0);

  parseRange(val: string): number {
    return parseInt(val);
  }

  chartCanvas = viewChild<ElementRef<HTMLCanvasElement>>('forecastChart');
  pidCanvas = viewChild<ElementRef<HTMLCanvasElement>>('pidForecastChart');
  kalmanCanvas = viewChild<ElementRef<HTMLCanvasElement>>('kalmanForecastChart');
  chart: any;
  pidChart: any;
  kalmanChart: any;
  chartData = signal<ForecastChart | null>(null);

  chartWidth = computed(() => {
    const data = this.chartData();
    if (!data || !data.labels) return '100%';
    const points = data.labels.length;
    return `${points * 22}px`;
  });

  constructor(private apiService: ApiService) { }

  ngOnInit() {
    this.fetchForecast();
  }

  onFileSelected(event: any) {
    const file = event.target.files[0];
    if (file) this.selectedFile.set(file);
  }

  runForecast() {
    const file = this.selectedFile();
    if (!file) return;

    this.isForecasting.set(true);
    this.forecastError.set(null);

    this.apiService.runForecast(file, this.historyWindow(), this.forecastQuantile).subscribe({
      next: () => {
        this.isForecasting.set(false);
        this.selectedFile.set(null);
        this.fetchForecast();
      },
      error: (err) => {
        this.isForecasting.set(false);
        const detail = err.error?.detail || 'Errore durante la generazione del forecast.';
        this.forecastError.set(detail);
      }
    });
  }

  fetchForecast() {
    this.isLoading.set(true);
    this.loadError.set(null);
    this.apiService.getForecastHistory().subscribe({
      next: (data: ForecastChart) => {
        this.hasData.set(true);
        this.isLoading.set(false);
        setTimeout(() => this.createChart(data), 0);
      },
      error: (err: any) => {
        this.isLoading.set(false);
        const detail = err.error?.detail || 'Errore nel recupero dei dati del forecast.';
        this.loadError.set(detail);
      }
    });
  }

  private createChart(data: ForecastChart) {
    const MAX_POINTS = 1400;
    this.dataCapped.set(false);

    let displayData = { ...data };
    if (data.labels.length > MAX_POINTS) {
      this.dataCapped.set(true);
      this.originalPointsCount.set(data.labels.length);
      const startIdx = data.labels.length - MAX_POINTS;
      displayData.labels = data.labels.slice(startIdx);
      displayData.actual = data.actual.slice(startIdx);
      displayData.prediction = data.prediction.slice(startIdx);
      displayData.prediction_p10 = data.prediction_p10.slice(startIdx);
      displayData.prediction_p90 = data.prediction_p90.slice(startIdx);
      if (data.actual_rounded) displayData.actual_rounded = data.actual_rounded.slice(startIdx);
      if (data.prediction_rounded) displayData.prediction_rounded = data.prediction_rounded.slice(startIdx);
      if (data.prediction_p10_rounded) displayData.prediction_p10_rounded = data.prediction_p10_rounded.slice(startIdx);
      if (data.prediction_p90_rounded) displayData.prediction_p90_rounded = data.prediction_p90_rounded.slice(startIdx);
      if (data.prediction_pid_rounded) displayData.prediction_pid_rounded = data.prediction_pid_rounded.slice(startIdx);
      if (data.prediction_kalman_rounded) displayData.prediction_kalman_rounded = data.prediction_kalman_rounded.slice(startIdx);
    }

    if (!displayData.labels || displayData.labels.length === 0) {
      this.hasData.set(false);
      return;
    }

    this.chartData.set(displayData);
    this.hasData.set(true);

    if (this.chart) this.chart.destroy();
    if (this.pidChart) this.pidChart.destroy();
    if (this.kalmanChart) this.kalmanChart.destroy();
    this.chart = null; this.pidChart = null; this.kalmanChart = null;

    // --- BRIDGE THE GAP ---
    let lastActualIndex = -1;
    for (let i = displayData.actual.length - 1; i >= 0; i--) {
      if (displayData.actual[i] !== null && displayData.actual[i] !== undefined) {
        lastActualIndex = i;
        break;
      }
    }

    if (lastActualIndex !== -1) {
      const lastVal = displayData.actual_rounded?.[lastActualIndex] ?? displayData.actual[lastActualIndex];
      if (lastVal !== null && lastVal !== undefined) {
        const val = Number(lastVal);
        const update = (arr: any[] | undefined, idx: number, v: number) => { if (arr && idx < arr.length) arr[idx] = v; };
        update(displayData.prediction, lastActualIndex, val);
        update(displayData.prediction_p10, lastActualIndex, val);
        update(displayData.prediction_p90, lastActualIndex, val);
        update(displayData.prediction_rounded, lastActualIndex, Math.round(val));
        update(displayData.prediction_p10_rounded, lastActualIndex, Math.round(val));
        update(displayData.prediction_p90_rounded, lastActualIndex, Math.round(val));
        update(displayData.prediction_pid_rounded, lastActualIndex, Math.round(val));
        update(displayData.prediction_kalman_rounded, lastActualIndex, Math.round(val));
      }
    }

    // --- RENDER ---
    const canvas = this.chartCanvas()?.nativeElement;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        this.chart = new Chart(ctx, {
          type: 'line',
          data: {
            labels: displayData.labels,
            datasets: [
              { label: 'P10', data: displayData.prediction_p10_rounded, borderColor: 'transparent', pointRadius: 0, fill: false, tension: 0.4 },
              { label: 'Banda Previsione', data: displayData.prediction_p90_rounded, borderColor: 'rgba(153, 102, 255, 0.5)', backgroundColor: 'rgba(153, 102, 255, 0.3)', fill: 0, tension: 0.4, borderWidth: 0, pointRadius: 0 },
              { label: 'Previsione', data: displayData.prediction_rounded, borderColor: 'rgba(153, 102, 255, 1)', fill: false, tension: 0.4, pointRadius: 2, borderWidth: 2 },
              { label: 'Reale', data: displayData.actual_rounded || displayData.actual, borderColor: 'rgba(75, 192, 192, 1)', backgroundColor: 'rgba(75, 192, 192, 0.2)', fill: false, tension: 0.4, pointRadius: 4 }
            ]
          },
          options: this.getChartOptions('Forecast (Modello Originale)')
        });
      }
    }

    const canvasPid = this.pidCanvas()?.nativeElement;
    if (canvasPid && displayData.prediction_pid_rounded) {
      const ctxPid = canvasPid.getContext('2d');
      if (ctxPid) {
        const actuals = displayData.actual_rounded || displayData.actual;
        this.pidChart = new Chart(ctxPid, {
          type: 'line',
          data: {
            labels: displayData.labels,
            datasets: [
              { label: 'P10', data: displayData.prediction_p10_rounded, borderColor: 'transparent', pointRadius: 0, fill: false, tension: 0.4 },
              { label: 'Banda Previsione', data: displayData.prediction_p90_rounded, backgroundColor: 'rgba(153, 102, 255, 0.3)', fill: 0, tension: 0.4, pointRadius: 0, borderWidth: 0 },
              { label: 'Allocazione PID', data: displayData.prediction_pid_rounded, borderColor: 'rgba(153, 102, 255, 1)', fill: false, tension: 0.4, pointRadius: 2, borderWidth: 2 },
              { label: 'Reale', data: actuals, borderColor: 'rgba(75, 192, 192, 1)', tension: 0.4, pointRadius: 4 }
            ]
          },
          options: this.getChartOptions('Forecast (Allocazione PID)')
        });
      }
    }

    const canvasKalman = this.kalmanCanvas()?.nativeElement;
    if (canvasKalman && displayData.prediction_kalman_rounded) {
      const ctxKalman = canvasKalman.getContext('2d');
      if (ctxKalman) {
        const actuals = displayData.actual_rounded || displayData.actual;
        this.kalmanChart = new Chart(ctxKalman, {
          type: 'line',
          data: {
            labels: displayData.labels,
            datasets: [
              { label: 'P10', data: displayData.prediction_p10_rounded, borderColor: 'transparent', pointRadius: 0, fill: false, tension: 0.4 },
              { label: 'Banda Previsione', data: displayData.prediction_p90_rounded, backgroundColor: 'rgba(153, 102, 255, 0.3)', fill: 0, tension: 0.4, pointRadius: 0, borderWidth: 0 },
              { label: 'Allocazione Kalman', data: displayData.prediction_kalman_rounded, borderColor: 'rgba(153, 102, 255, 1)', fill: false, tension: 0.4, pointRadius: 2, borderWidth: 2 },
              { label: 'Reale', data: actuals, borderColor: 'rgba(75, 192, 192, 1)', tension: 0.4, pointRadius: 4 }
            ]
          },
          options: this.getChartOptions('Forecast (Allocazione Kalman)')
        });
      }
    }
  }

  private getChartOptions(title: string): any {
    return {
      devicePixelRatio: window.devicePixelRatio || 2,
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, title: { display: true, text: title, color: '#ffffff' } },
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#cccccc' },
          title: { display: true, text: 'Capacità (Istanze)', color: '#ffffff', font: { size: 16, weight: 'bold', family: 'Inter' }, padding: { bottom: 20 } }
        },
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#cccccc', maxRotation: 45, minRotation: 45 },
          title: { display: true, text: 'Tempo', color: '#ffffff', font: { size: 16, weight: 'bold', family: 'Inter' }, padding: { top: 20 } }
        }
      }
    };
  }
}
