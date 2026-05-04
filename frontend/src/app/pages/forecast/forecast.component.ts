import { Component, OnInit, ElementRef, signal, viewChild } from '@angular/core';
import { ApiService } from '../../services/api.service';
import { ForecastChart } from '../../models/api-data.model';
import { Chart } from 'chart.js/auto';

@Component({
  selector: 'app-forecast',
  standalone: true,
  imports: [],
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
      </div>

      @if (hasData()) {
        <div class="legend-custom">
          <div class="legend-item"><span class="dot actual"></span> Dati Reali</div>
          <div class="legend-item"><span class="dot p50"></span> Previsione (P50)</div>
          <div class="legend-item"><span class="dot p90"></span> Area di Previsione (P10-P90)</div>
        </div>
      }

      <div class="chart-container">
        @if (isLoading()) {
          <div class="flex-row" style="height: 100%; justify-content: center;">
             <div class="spinner"></div>
             <span class="text-primary">Caricamento grafico...</span>
          </div>
        }
        
        <div class="chart-wrapper" [hidden]="isLoading() || !hasData()">
          <canvas #forecastChart></canvas>
        </div>
      </div>
    </div>
  `,
  styles: []
})
export class ForecastComponent implements OnInit {
  isLoading = signal(true);
  isForecasting = signal(false);
  hasData = signal(false);
  selectedFile = signal<File | null>(null);
  
  chartCanvas = viewChild<ElementRef<HTMLCanvasElement>>('forecastChart');
  chart: any;

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
    this.apiService.runForecast(file).subscribe({
      next: (res: any) => {
        console.log('Forecast completato:', res);
        this.isForecasting.set(false);
        this.selectedFile.set(null);
        this.fetchForecast(); // Ricarica il grafico con i nuovi dati
      },
      error: (err) => {
        console.error('Errore durante il forecast:', err);
        alert('Errore durante la generazione del forecast.');
        this.isForecasting.set(false);
      }
    });
  }

  fetchForecast() {
    this.isLoading.set(true);
    this.apiService.getForecastHistory().subscribe({
      next: (data: ForecastChart) => {
        this.hasData.set(true);
        this.isLoading.set(false);
        setTimeout(() => this.createChart(data), 0);
      },
      error: (err: any) => {
        console.error('Errore dati forecast:', err);
        this.isLoading.set(false);
      }
    });
  }

  private createChart(data: ForecastChart) {
    const canvas = this.chartCanvas()?.nativeElement;
    if (!canvas) return;

    // --- BRIDGE THE GAP (FRONTEND) ---
    // Trova l'ultimo punto storico reale
    let lastActualIndex = -1;
    for (let i = data.actual.length - 1; i >= 0; i--) {
      if (data.actual[i] !== null && data.actual[i] !== undefined) {
        lastActualIndex = i;
        break;
      }
    }

    if (lastActualIndex !== -1) {
      const lastActualValue = data.actual_rounded?.[lastActualIndex] ?? data.actual[lastActualIndex];
      
      if (lastActualValue !== null && lastActualValue !== undefined) {
        const bridgeVal = Number(lastActualValue);
        
        // Assicuriamoci che gli array esistano e siano della lunghezza corretta
        const updateVal = (arr: any[] | undefined, idx: number, val: number) => {
          if (arr && idx >= 0 && idx < arr.length) {
            arr[idx] = val;
          }
        };

        updateVal(data.prediction, lastActualIndex, bridgeVal);
        updateVal(data.prediction_p10, lastActualIndex, bridgeVal);
        updateVal(data.prediction_p90, lastActualIndex, bridgeVal);
        updateVal(data.prediction_rounded, lastActualIndex, Math.round(bridgeVal));
        updateVal(data.prediction_p10_rounded, lastActualIndex, Math.round(bridgeVal));
        updateVal(data.prediction_p90_rounded, lastActualIndex, Math.round(bridgeVal));
      }
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (this.chart) this.chart.destroy();

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [
          {
            label: 'P10',
            data: data.prediction_p10_rounded || data.prediction_p10,
            borderColor: 'rgba(255, 99, 132, 0)', // Trasparente
            pointRadius: 0,
            fill: false,
            tension: 0.4
          },
          {
            label: 'Area di Previsione (P10-P90)',
            data: data.prediction_p90_rounded || data.prediction_p90,
            borderColor: 'rgba(255, 99, 132, 0.5)',
            backgroundColor: 'rgba(255, 99, 132, 0.25)',
            fill: 0, // Riempie verso il dataset index 0 (P10)
            tension: 0.4,
            borderWidth: 1,
            cubicInterpolationMode: 'monotone',
            pointRadius: 2
          },
          {
            label: 'Previsione (P50)',
            data: data.prediction_rounded || data.prediction,
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
            data: data.actual_rounded || data.actual,
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
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' },
            title: { display: true, text: 'Capacità (Istanze)', color: '#ffffff' }
          },
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', maxRotation: 45, minRotation: 45 },
            title: { display: true, text: 'Tempo', color: '#ffffff' }
          }
        }
      }
    });
  }
}
