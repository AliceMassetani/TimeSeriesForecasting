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

      <!-- ERRORE CARICAMENTO DATI -->
      @if (loadError()) {
        <div class="alert-box error">
          <span class="alert-icon">✕</span>
          <div class="alert-content">
            <p>{{ loadError() }}</p>
          </div>
        </div>
      }

      <!-- AVVISO DATI TAGLIATI -->
      @if (dataCapped()) {
        <div class="alert-box warning">
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

      @if (hasData()) {
        <!-- GRAFICO ORIGINALE -->
        <div class="page-header" style="margin-top: 2rem;">
          <h3>Risultati Modello Originale (TSMixer)</h3>
          <p class="subtitle">Dati grezzi di previsione senza correzioni.</p>
        </div>

        <div class="legend-custom">
          <div class="legend-item"><span class="dot actual"></span> Dati Reali</div>
          <div class="legend-item"><span class="dot p50"></span> Previsione (P50)</div>
          <div class="legend-item"><span class="dot p90"></span> Area di Previsione (P10-P90)</div>
        </div>

        <div class="chart-container">
          <div class="chart-scroll-container">
            <div class="chart-wrapper" [style.width]="chartWidth()" [style.height.px]="chartHeight()">
              <canvas #forecastChart></canvas>
            </div>
          </div>
        </div>

        @if (chartData()?.prediction_pid_rounded) {
          <div class="divider" style="margin: 2rem 0; border-top: 1px solid rgba(255,255,255,0.05);"></div>

          <div class="page-header">
            <h3 class="text-pid" style="color: #9966ff;">Allocazione Ottimizzata (Correzione Adattiva)</h3>
            <p class="subtitle">Piano di allocazione corretto tramite analisi del bias e dei picchi recenti.</p>
          </div>

          <div class="legend-custom" style="margin-top: 1rem;">
            <div class="legend-item"><span class="dot actual"></span> Dati Reali</div>
            <div class="legend-item"><span class="dot p50" style="background: rgba(153, 102, 255, 1)"></span> Allocazione Corretta</div>
            <div class="legend-item"><span class="dot p90" style="background: rgba(153, 102, 255, 0.3)"></span> Banda Previsione</div>
          </div>

          <div class="chart-container">
            <div class="chart-scroll-container">
              <div class="chart-wrapper" [style.width]="chartWidth()" [style.height.px]="chartHeight()">
                <canvas #pidForecastChart></canvas>
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
  chart: any;
  pidChart: any;
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
    if (file) {
      this.selectedFile.set(file);
    }
  }

  runForecast() {
    const file = this.selectedFile();
    if (!file) return;

    this.isForecasting.set(true);
    this.forecastError.set(null);
    this.dataCapped.set(false);

    this.apiService.runForecast(file, this.historyWindow(), this.forecastQuantile).subscribe({
      next: (res: any) => {
        console.log('Forecast completato:', res);
        this.isForecasting.set(false);
        this.selectedFile.set(null);
        this.fetchForecast(); // Ricarica il grafico con i nuovi dati
      },
      error: (err) => {
        console.error('Errore durante il forecast:', err);
        this.isForecasting.set(false);
        const detail = err.error?.detail || 'Errore durante la generazione del forecast.';
        this.forecastError.set(detail);
      }
    });
  }

  fetchForecast() {
    this.isLoading.set(true);
    this.loadError.set(null);
    this.dataCapped.set(false);
    this.apiService.getForecastHistory().subscribe({
      next: (data: ForecastChart) => {
        this.hasData.set(true);
        this.isLoading.set(false);
        setTimeout(() => this.createChart(data), 0);
      },
      error: (err: any) => {
        console.error('Errore dati forecast:', err);
        this.isLoading.set(false);
        const detail = err.error?.detail || 'Errore nel recupero dei dati del forecast.';
        this.loadError.set(detail);
      }
    });
  }

  private createChart(data: ForecastChart) {
    // --- LIMITE TECNICO CANVAS ---
    // I browser non supportano canvas più larghi di ~32k pixel.
    // Con 22px per punto, il limite è circa 1400-1500 punti.
    const MAX_POINTS = 1400;

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
    }

    if (!displayData.labels || displayData.labels.length === 0) {
      this.hasData.set(false);
      return;
    }

    this.chartData.set(displayData);
    this.hasData.set(true);
    const canvas = this.chartCanvas()?.nativeElement;
    if (!canvas) return;

    // --- BRIDGE THE GAP (FRONTEND) ---
    // Usiamo displayData invece di data
    let lastActualIndex = -1;
    for (let i = displayData.actual.length - 1; i >= 0; i--) {
      if (displayData.actual[i] !== null && displayData.actual[i] !== undefined) {
        lastActualIndex = i;
        break;
      }
    }

    if (lastActualIndex !== -1) {
      const lastActualValue = displayData.actual_rounded?.[lastActualIndex] ?? displayData.actual[lastActualIndex];

      if (lastActualValue !== null && lastActualValue !== undefined) {
        const bridgeVal = Number(lastActualValue);

        const updateVal = (arr: any[] | undefined, idx: number, val: number) => {
          if (arr && idx >= 0 && idx < arr.length) {
            arr[idx] = val;
          }
        };

        updateVal(displayData.prediction, lastActualIndex, bridgeVal);
        updateVal(displayData.prediction_p10, lastActualIndex, bridgeVal);
        updateVal(displayData.prediction_p90, lastActualIndex, bridgeVal);
        updateVal(displayData.prediction_rounded, lastActualIndex, Math.round(bridgeVal));
        updateVal(displayData.prediction_p10_rounded, lastActualIndex, Math.round(bridgeVal));
        updateVal(displayData.prediction_p90_rounded, lastActualIndex, Math.round(bridgeVal));
        updateVal(displayData.prediction_pid_rounded, lastActualIndex, Math.round(bridgeVal));
      }
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (this.chart) this.chart.destroy();
    if (this.pidChart) this.pidChart.destroy();

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: displayData.labels,
        datasets: [
          {
            label: 'P10',
            data: displayData.prediction_p10_rounded || displayData.prediction_p10,
            borderColor: 'rgba(255, 99, 132, 0)', // Trasparente
            pointRadius: 0,
            fill: false,
            tension: 0.4
          },
          {
            label: 'Area di Previsione (P10-P90)',
            data: displayData.prediction_p90_rounded || displayData.prediction_p90,
            borderColor: 'rgba(153, 102, 255, 0.5)',
            backgroundColor: 'rgba(153, 102, 255, 0.3)',
            fill: 0, // Riempie verso il dataset index 0 (P10)
            tension: 0.4,
            borderWidth: 0,
            cubicInterpolationMode: 'monotone',
            pointRadius: 0
          },
          {
            label: 'Previsione (P50)',
            data: displayData.prediction_rounded || displayData.prediction,
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0)',
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 2,
            borderWidth: 2
          },
          {
            label: 'Dati Reali',
            data: displayData.actual_rounded || displayData.actual,
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 4,
            pointBackgroundColor: 'rgba(75, 192, 192, 1)'
          }
        ]
      },
      options: {
        devicePixelRatio: window.devicePixelRatio || 2,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: {
            beginAtZero: true,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' },
            title: {
              display: true,
              text: 'Capacità (Istanze)',
              color: '#ffffff',
              font: { size: 16, weight: 'bold', family: 'Inter' },
              padding: { bottom: 20 }
            }
          },
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', maxRotation: 45, minRotation: 45 },
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

    // Sincronizzazione Scale Y Globale
    this.chart.update('none');
    let yMin = this.chart.scales['y'].min;
    let yMax = this.chart.scales['y'].max;

    // 2. Chart PID Forecast
    const canvasPid = this.pidCanvas()?.nativeElement;
    if (canvasPid && displayData.prediction_pid_rounded) {
      const ctxPid = canvasPid.getContext('2d');
      if (ctxPid) {
        const diffPid = (displayData.prediction_pid_rounded || []).map((val, i) => val - (displayData.actual_rounded?.[i] || displayData.actual?.[i] || 0));

        this.pidChart = new Chart(ctxPid, {
          type: 'line',
          data: {
            labels: displayData.labels,
            datasets: [
              { label: 'P10', data: displayData.prediction_p10_rounded, pointRadius: 0, fill: false, tension: 0.4, borderColor: 'transparent' },
              { label: 'Banda Previsione', data: displayData.prediction_p90_rounded, fill: 0, tension: 0.4, pointRadius: 0, backgroundColor: 'rgba(153, 102, 255, 0.3)' },
              { label: 'Allocazione Corretta', data: displayData.prediction_pid_rounded, borderColor: 'rgba(153, 102, 255, 1)', tension: 0.4, pointRadius: 0, borderWidth: 2 },
              { label: 'Diff (PID)', data: diffPid, borderColor: 'rgba(255, 159, 64, 1)', backgroundColor: 'rgba(255, 159, 64, 0.1)', fill: true, tension: 0.4, pointRadius: 0 },
              { label: 'Actual', data: displayData.actual_rounded, borderColor: 'rgba(75, 192, 192, 1)', tension: 0.4, pointRadius: 4 }
            ]
          },
          options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } } }
        });

        this.pidChart.update('none');
        const yAxisPid = this.pidChart.scales['y'];
        yMin = Math.min(yMin, yAxisPid.min);
        yMax = Math.max(yMax, yAxisPid.max);
        
        // Aggiungiamo un buffer del 15% per evitare tagli in alto
        yMax = yMax * 1.15;

        // Regoliamo l'altezza del grafico in base al range
        const baseHeight = 400;
        const dynamicHeight = Math.max(baseHeight, Math.min(800, yMax * 8)); 
        this.chartHeight.set(dynamicHeight);

        // Riapplichiamo la scala globale ad ENTRAMBI
        this.chart.options.scales.y.min = yMin;
        this.chart.options.scales.y.max = yMax;
        this.chart.update();

        this.pidChart.options = {
          devicePixelRatio: window.devicePixelRatio || 2,
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            title: { display: true, text: 'Forecast (Correzione Adattiva)', color: '#ffffff' }
          },
          scales: {
            y: {
              min: yMin,
              max: yMax,
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#94a3b8' },
              title: { display: true, text: 'Capacità (Istanze)', color: '#ffffff', font: { size: 16, weight: 'bold' }, padding: { bottom: 20 } }
            },
            x: {
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#94a3b8', maxRotation: 45 },
              title: { display: true, text: 'Tempo', color: '#ffffff', font: { size: 16, weight: 'bold' }, padding: { top: 20 } }
            }
          }
        };
        this.pidChart.update();
      }
    }
  }
}
