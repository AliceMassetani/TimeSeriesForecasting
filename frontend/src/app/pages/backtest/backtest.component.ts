import { Component, ElementRef, OnInit, signal, viewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { Chart } from 'chart.js/auto';
import { DecimalPipe } from '@angular/common';

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
      </div>

      <!-- GRAFICO E LEGENDA -->
      @if (hasData()) {
        <div class="legend-custom">
          <div class="legend-item"><span class="dot actual"></span> Reale</div>
          <div class="legend-item"><span class="dot p50"></span> Previsione</div>
          <div class="legend-item"><span class="dot p70"></span> Previsione P70</div>
          <div class="legend-item"><span class="dot diff"></span> Differenza</div>
        </div>
      }

      <div class="chart-container" [hidden]="!hasData()">
        <div class="chart-wrapper">
          <canvas #backtestChart></canvas>
        </div>
      </div>
      
      <!-- METRICHE -->
      @if (metrics()) {
        <div class="metrics-grid">
          <div class="metrics-card">
            <span class="label">MSE</span>
            <span class="value">{{ metrics()?.mse | number: '1.3-3' }}</span>
          </div>        
          <div class="metrics-card">
            <span class="label">RMSE</span>
            <span class="value">{{ metrics()?.rmse | number: '1.3-3' }}</span>
          </div> 
          <div class="metrics-card"> 
            <span class="label">MAE</span>
            <span class="value">{{ metrics()?.mae | number: '1.3-3' }}</span>
          </div>
          <div class="metrics-card">
            <span class="label">R²</span>
            <span class="value">{{ metrics()?.r2 | number: '1.3-3' }}</span>
          </div>        
        </div>
      }
    </div>
  `,
  styles: [] 
})
export class BacktestComponent implements OnInit {
  selectedFile: File | null = null;
  limit?: number;
  startDate?: string;
  endDate?: string;

  isBacktesting = signal(false);
  isLoadingChart = signal(false);
  hasData = signal(false);
  metrics = signal<any>(null);

  chartCanvas = viewChild<ElementRef<HTMLCanvasElement>>('backtestChart');
  chart: any;

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
    
    this.apiService.runBacktest(this.selectedFile).subscribe({
      next: (response: any) => {
        console.log('--- BACKTEST COMPLETATO CON SUCCESSO ---', response);
        this.metrics.set(response.metrics);
        this.loadBacktestChart();
        this.isBacktesting.set(false);
      },
      error: (err: any) => {
        console.error('--- ERRORE BACKTEST ---', err);
        this.isBacktesting.set(false);
        alert('Errore durante il backtest.');
      }
    });
  }

  loadBacktestChart() {
    console.log('--- CARICAMENTO STORICO BACKTEST ---');
    this.isLoadingChart.set(true);
    
    this.apiService.getBacktestHistory(this.limit, this.startDate, this.endDate).subscribe({
      next: (data: any) => {
        console.log('--- STORICO CARICATO ---', data);
        this.metrics.set(data.metrics);
        this.createChart(data);
        this.isLoadingChart.set(false);
      },
      error: (err: any) => {
        console.error('--- ERRORE CARICAMENTO STORICO ---', err);
        this.isLoadingChart.set(false);
      }
    });
  }

  private createChart(data: any) {
    this.hasData.set(true);
    const canvas = this.chartCanvas()?.nativeElement;
    if (!canvas) return;

    const context = canvas.getContext('2d');
    if (!context) return;

    if (this.chart) this.chart.destroy();

    this.chart = new Chart(context, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [
          {
            label: 'Actual',
            data: data.actual_rounded,
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 4
          },
          {
            label: 'Prediction',
            data: data.prediction_rounded,
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 2
          },
          {
            label: 'Diff',
            data: data.diff_rounded_instances,
            borderColor: 'rgba(255, 159, 64, 1)',
            backgroundColor: 'rgba(255, 159, 64, 0.2)',
            fill: true,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 2
          },
          {
            label: 'Prediction P70',
            data: data.prediction_p70_rounded,
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 2
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
          x: {
            ticks: { color: '#cccccc', maxRotation: 45, minRotation: 45 },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          },
          y: {
            ticks: { color: '#cccccc' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          }
        }
      }
    });
  }
}
